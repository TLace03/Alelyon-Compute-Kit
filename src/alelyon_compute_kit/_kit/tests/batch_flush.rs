//! Deferred submission (increment 5): the flush points, the barrier rule, the
//! descriptor arena and the retirement list, on a real device.
//!
//! `Context::dispatch` RECORDS; the work is submitted at the next flush, and
//! `batch::FlushReason` is the whole list of reasons a flush happens. These
//! tests fire SEVEN of the eight wired reasons -- `Readback`, `Upload`,
//! `ExplicitSync`, `TimedDispatch`, `BatchFull`, `DescriptorsExhausted` and
//! `RetiredBytes` -- and assert that each moved its own counter and no other,
//! so those cannot drift into a stale sentence in a document.
//!
//! `Close` is the eighth and its counter is unreachable from here, not
//! untested by choice: `Context::flush` is `pub(crate)`, the only public
//! wrapper (`sync`) hardcodes `ExplicitSync`, and no C entry exposes
//! `flushes_by_reason`, so neither an integration test nor a Python test can
//! ask for a `Close` flush or read its counter. What is pinned instead is the
//! BRANCH, in `a_close_with_work_recorded_flushes_and_reports_the_failure`:
//! only a `Close` flush of a non-empty batch can put that message in
//! `ack_last_error`. Its success path has no observable at ABI 9 and is
//! UNMEASURED.
//!
//! Two of them are FALSIFIERS, not controls. They run the same fixture twice
//! in one session, differing in one input -- the barrier rule, or the
//! descriptor arena -- and require the sabotaged arm to DIVERGE. If it does
//! not, the test reports UNMEASURED and does not pass as evidence: a sabotage
//! that never applied reads as a passing row.
//!
//! Needs a Vulkan device. Without one every test here prints UNMEASURED and
//! returns, which is a skip with its reason, never a pass; set
//! `ACK_REQUIRE_DEVICE=1` to turn that into a failure on a machine that is
//! supposed to have the device. A GREEN RUN WITHOUT THAT VARIABLE IS NOT
//! DEVICE EVIDENCE.

use alelyon_compute_kit::batch::{
    flush_reason_index, flush_reason_name, is_wired, FlushReason, ALL_FLUSH_REASONS,
    FLUSH_REASON_COUNT,
};
use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, ELEMENTS_PER_GROUP, OPERANDS,
    PUSH_BYTES,
};
use alelyon_compute_kit::{Access, AckError, Buffer, Context, Kernel};

const POINTWISE_F32: &[u8] = include_bytes!("../kernels/pointwise_f32.spv");
const CAST_F32_BF16: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");

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

/// Assert that exactly `reason` moved, by `by`, and nothing else did.
fn only(
    before: [u64; FLUSH_REASON_COUNT],
    after: [u64; FLUSH_REASON_COUNT],
    reason: FlushReason,
    by: u64,
) {
    for r in ALL_FLUSH_REASONS {
        let i = flush_reason_index(r);
        let expected = before[i] + if r == reason { by } else { 0 };
        assert_eq!(
            after[i],
            expected,
            "{}: expected {} flush(es) of {}, saw {}",
            flush_reason_name(reason),
            expected,
            flush_reason_name(r),
            after[i]
        );
    }
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

/// A `Copy` of `src` into `dst` over `shape`, `src` seen through `src_view`.
/// The unused `y` and `z` slots are bound to the output with the output's own
/// view, which is the plan's own rule for a slot the operation does not read.
fn copy_plan(
    shape: &[u64],
    src: &Buffer,
    src_view: StridedView,
    dst: &Buffer,
) -> (PointwisePlan, [u8; PUSH_BYTES]) {
    let out_view = contiguous(shape);
    let views: [StridedView; OPERANDS] = [src_view, out_view, out_view, out_view];
    let storage: [Storage; OPERANDS] = [
        storage_of(src),
        storage_of(dst),
        storage_of(dst),
        storage_of(dst),
    ];
    let plan = PointwisePlan::new(PointwiseOp::Copy, shape, views, Scalars::new(0.0), storage)
        .expect("copy plan");
    let push = plan.push_constants();
    (plan, push)
}

fn pointwise_kernel(ctx: &Context) -> Kernel {
    ctx.kernel(POINTWISE_F32, OPERANDS as u32, PUSH_BYTES as u32)
        .expect("pointwise kernel")
}

fn access_for(src: &Buffer, dst: &Buffer) -> [Access; OPERANDS] {
    // slot order: x, y, z, out. y and z are bound to the output and the
    // operation reads neither; an unused slot is declared Read.
    [
        Access::read(src.id()),
        Access::read(dst.id()),
        Access::read(dst.id()),
        Access::write(dst.id()),
    ]
}

#[test]
fn a_fresh_context_has_recorded_nothing_and_flushed_nothing() {
    let Some(ctx) = open_or_unmeasured("fresh context") else {
        return;
    };
    assert_eq!(ctx.recorded_items(), 0);
    assert_eq!(ctx.recorded_barriers(), 0);
    assert_eq!(ctx.flushes(), 0);
    assert_eq!(
        ctx.live_objects(),
        0,
        "the batch's command buffer and fence are allocated at the first recorded item, not at open"
    );
    assert_eq!(ctx.flushes_by_reason(), [0u64; FLUSH_REASON_COUNT]);
}

#[test]
fn a_dispatch_is_recorded_and_the_readback_is_what_submits_it() {
    let Some(ctx) = open_or_unmeasured("record then readback") else {
        return;
    };
    let n = 4096u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");

    let before = ctx.flushes_by_reason();
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    assert_eq!(ctx.recorded_items(), 1, "recorded, not run");
    only(before, ctx.flushes_by_reason(), FlushReason::Readback, 0);

    let before = ctx.flushes_by_reason();
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    only(before, ctx.flushes_by_reason(), FlushReason::Readback, 1);
    assert_eq!(ctx.recorded_items(), 0, "the batch was submitted");
    assert_eq!(back, values, "the recorded dispatch ran at the flush");

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn every_wired_flush_reason_moves_its_own_counter_and_no_other() {
    let Some(ctx) = open_or_unmeasured("flush reasons") else {
        return;
    };
    let n = 4096u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
    let bytes = f32_bytes(&values);
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    let record = |ctx: &Context| {
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
            .expect("record")
    };

    // Upload: its own fenced submission (the default), and it must not run
    // ahead of a recorded dispatch. The RECORDED upload (2026-09-10, opt-in)
    // fires no flush at all and has its own test below.
    assert!(!ctx.deferred_upload(), "the default upload is eager");
    ctx.upload(&src, &bytes).expect("prime");
    record(&ctx);
    let before = ctx.flushes_by_reason();
    ctx.upload(&src, &bytes).expect("upload");
    only(before, ctx.flushes_by_reason(), FlushReason::Upload, 1);

    // Readback
    record(&ctx);
    let before = ctx.flushes_by_reason();
    let _ = ctx.download(&dst).expect("download");
    only(before, ctx.flushes_by_reason(), FlushReason::Readback, 1);

    // ExplicitSync
    record(&ctx);
    let before = ctx.flushes_by_reason();
    ctx.sync().expect("sync");
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::ExplicitSync,
        1,
    );

    // an empty batch submits nothing and is not counted
    let before = ctx.flushes_by_reason();
    ctx.sync().expect("sync of an empty batch");
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::ExplicitSync,
        0,
    );

    // TimedDispatch: it flushes what is recorded, then times its own work
    record(&ctx);
    let before = ctx.flushes_by_reason();
    let ms = ctx
        .dispatch_timed(&kernel, &push, plan.dispatch_groups(), 1)
        .expect("timed dispatch");
    assert!(ms >= 0.0, "a device duration, not a batch's: {ms}");
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::TimedDispatch,
        1,
    );

    // BatchFull, at exactly the budget's item
    ctx.set_max_recorded_items(3);
    let before = ctx.flushes_by_reason();
    record(&ctx);
    record(&ctx);
    only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 0);
    assert_eq!(ctx.recorded_items(), 2);
    record(&ctx);
    only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 1);
    assert_eq!(ctx.recorded_items(), 0, "the third item filled the batch");
    ctx.set_max_recorded_items(alelyon_compute_kit::batch::MAX_RECORDED_ITEMS);

    // the accounting reason is declared and unwired: nothing can move it
    let counts = ctx.flushes_by_reason();
    for r in ALL_FLUSH_REASONS {
        if !is_wired(r) {
            assert_eq!(
                counts[flush_reason_index(r)],
                0,
                "{} is unwired and must never fire",
                flush_reason_name(r)
            );
        }
    }

    ctx.sync().expect("drain");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn the_batch_records_one_barrier_between_every_pair_of_adjacent_items() {
    let Some(ctx) = open_or_unmeasured("barrier count") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![1.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    for i in 1..=8u32 {
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
            .expect("record");
        assert_eq!(ctx.recorded_items(), i);
        assert_eq!(
            ctx.recorded_barriers(),
            i - 1,
            "barriers must be items - 1; no barrier before the first item"
        );
    }
    ctx.sync().expect("sync");
    assert_eq!(
        ctx.recorded_barriers(),
        0,
        "a flushed batch records nothing"
    );
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn descriptors_exhausted_flushes_and_the_retry_after_it_succeeds() {
    // The arena is `sets_per_kernel_pool` sets deep; a batch that records more
    // dispatches of one kernel than that must flush and continue, not refuse.
    // The depth is read from the environment ONCE AT OPEN, so this test drives
    // the branch by recording that many dispatches rather than by setting an
    // environment variable -- the test harness runs these in threads of one
    // process, and a process-wide variable would reach into whatever other
    // test happened to be opening a context at the time.
    let Some(ctx) = open_or_unmeasured("descriptor arena") else {
        return;
    };
    let sets = ctx.sets_per_kernel_pool();
    assert!(sets >= 1);
    // the batch depth must not be the thing that flushes first: neither the
    // depth itself nor the ramp's short first batches after the upload
    ctx.set_max_recorded_items(sets * 4);
    ctx.set_batch_ramp(false);
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    assert_eq!(kernel.descriptor_sets(), sets);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");

    let before = ctx.flushes_by_reason();
    for _ in 0..sets {
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
            .expect("record");
    }
    assert_eq!(ctx.descriptor_watermark(&kernel), sets);
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::DescriptorsExhausted,
        0,
    );
    // the next one finds the arena exhausted, flushes, and records
    ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect("record after the arena is exhausted");
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::DescriptorsExhausted,
        1,
    );
    assert_eq!(ctx.recorded_items(), 1, "the retry recorded");
    assert_eq!(ctx.descriptor_watermark(&kernel), 1);
    assert_eq!(from_bytes(&ctx.download(&dst).expect("download")), values);
    ctx.set_max_recorded_items(alelyon_compute_kit::batch::MAX_RECORDED_ITEMS);
    ctx.set_batch_ramp(true);

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn a_buffer_freed_while_a_batch_is_open_is_retired_and_destroyed_by_the_flush() {
    let Some(ctx) = open_or_unmeasured("retirement") else {
        return;
    };
    // THE UN-POOLED ARM, AND THE NUMBERS BELOW ARE ITS ORIGINAL ONES. What
    // this test pins is the RETIREMENT boundary: a pair freed while a batch is
    // open is not released until the flush. The buffer pool does not move that
    // boundary -- it is the pool's own entry point -- but it does change what
    // "released" costs, because a flush now hands the pair to the pool instead
    // of destroying it, so the delta across the flush is 2 objects and not 4.
    // Turning the pool off here keeps this test asserting the object law it
    // was written for, unchanged; the pooled counterpart is
    // `a_buffer_freed_while_a_batch_is_open_reaches_the_pool_only_at_the_flush`
    // in tests/buffer_pool.rs, which asserts the same boundary through the
    // pool's own return counter.
    ctx.set_buffer_pool_enabled(false);
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let doomed = ctx.buffer(n * 4).expect("doomed");
    let doomed_id = doomed.id();
    ctx.upload(&src, &f32_bytes(&vec![2.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");

    // Sampled BEFORE the first recorded item, so the batch's own command
    // buffer and fence are inside the delta rather than smuggled into the
    // baseline. A flush releases FOUR objects, not two -- the retired buffer
    // and its memory, and the submission's fence and command buffer -- and a
    // baseline taken with the batch already open hides the second pair.
    let before_batch = ctx.live_objects();
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");

    let open = ctx.live_objects();
    assert_eq!(
        open,
        before_batch + 2,
        "the first recorded item allocates the batch's command buffer and fence"
    );
    assert_eq!(ctx.recorded_items(), 1, "a batch is open");
    let torn_before = ctx.submissions_torn_down();
    ctx.destroy_buffer(doomed);
    assert_eq!(
        ctx.live_objects(),
        open,
        "recorded work may still name it: the pair is retired, not destroyed, and stays counted"
    );
    ctx.sync().expect("flush");
    assert_eq!(
        ctx.live_objects(),
        open - 4,
        "across the flush: the retired buffer and its memory (2), and the batch's own fence and command buffer (2)"
    );
    assert_eq!(
        ctx.live_objects(),
        before_batch - 2,
        "net over the whole batch: only the freed buffer's pair is gone, the submission's objects came and went"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 1,
        "one submission released per flush, not one per recorded operation"
    );

    // with no batch open the free is immediate, as it always was
    let again = ctx.buffer(n * 4).expect("again");
    assert_ne!(again.id(), doomed_id);
    let before = ctx.live_objects();
    assert_eq!(ctx.recorded_items(), 0, "no batch is open");
    ctx.destroy_buffer(again);
    assert_eq!(ctx.live_objects(), before - 2, "destroyed, not retired");

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn a_dispatch_that_declares_buffers_the_kernel_is_not_bound_to_is_refused_by_name() {
    let Some(ctx) = open_or_unmeasured("declared access") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![3.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    let groups = plan.dispatch_groups();

    // a slot short
    let short = [Access::read(src.id()), Access::read(dst.id())];
    let err = ctx
        .dispatch(&kernel, &push, groups, 1, &short)
        .expect_err("a short declaration is refused");
    assert!(matches!(err, AckError::Unsupported(_)), "{err}");
    assert!(format!("{err}").contains("bound to 4"), "{err}");

    // the right count, the wrong buffer in one slot
    let wrong = [
        Access::read(dst.id()),
        Access::read(dst.id()),
        Access::read(dst.id()),
        Access::write(dst.id()),
    ];
    let err = ctx
        .dispatch(&kernel, &push, groups, 1, &wrong)
        .expect_err("a wrong slot is refused");
    assert!(format!("{err}").contains("slot 0"), "{err}");

    // and the refusal precedes the recording
    assert_eq!(
        ctx.recorded_items(),
        0,
        "a refused dispatch records nothing"
    );
    ctx.dispatch(&kernel, &push, groups, 1, &access_for(&src, &dst))
        .expect("the correct declaration records");
    assert_eq!(ctx.recorded_items(), 1);

    ctx.sync().expect("drain");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn the_eager_arm_and_the_deferred_arm_agree_bit_for_bit() {
    // The rule this increment is held to: it changes WHEN work is submitted,
    // not WHAT is computed. The two arms are the same code path with one
    // number changed (`max_recorded_items`), alternated inside one session on
    // one context, so nothing but the batch depth differs.
    let Some(ctx) = open_or_unmeasured("eager vs deferred") else {
        return;
    };
    let rows = 64u64;
    let cols = 64u64;
    let n = rows * cols;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    // a transposed read, so the answer depends on more than a straight copy
    let shape = [cols, rows];
    let transposed = StridedView::new(0, [1, cols, 0, 0]);
    let (plan, push) = copy_plan(&shape, &src, transposed, &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");

    let mut arms: Vec<(u32, Vec<f32>)> = Vec::new();
    for round in 0..4 {
        for depth in [1u32, 64] {
            ctx.set_max_recorded_items(depth);
            ctx.upload(&dst, &f32_bytes(&vec![f32::NAN; n as usize]))
                .expect("clear");
            for _ in 0..8 {
                ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
                    .expect("dispatch");
            }
            let out = from_bytes(&ctx.download(&dst).expect("download"));
            assert!(
                out.iter().all(|x| x.is_finite()),
                "round {round}, depth {depth}: the clear was not overwritten"
            );
            arms.push((depth, out));
        }
    }
    ctx.set_max_recorded_items(alelyon_compute_kit::batch::MAX_RECORDED_ITEMS);
    let reference = &arms[0].1;
    for (depth, out) in &arms {
        assert_eq!(
            out, reference,
            "depth {depth} produced different bits: this increment changes when work is submitted, not what it computes"
        );
    }
    // and the reference is the transpose it claims to be
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            assert_eq!(
                reference[c * rows as usize + r],
                values[r * cols as usize + c]
            );
        }
    }

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn two_dispatches_of_one_kernel_in_one_batch_each_take_their_own_descriptor_set() {
    // The wrong-answer hazard deferral introduces, and its falsifier. Before
    // this increment a kernel had ONE descriptor set and `bind` rewrote it;
    // that was safe only because each dispatch was submitted and waited for
    // before the next bind. Recording two dispatches of one kernel into one
    // batch falsifies that reason.
    //
    // THE CONTROL ARM IS ALSO THE GUARD. It asserts both answers exactly, and
    // it does so with no feature enabled: removing the arena from production
    // (making every dispatch of a kernel take `sets[0]`, which is precisely
    // what the sabotage does) makes `got_b == ones` fail here, on a plain
    // `cargo test` with a device. The arena is not left to the opt-in arm.
    //
    // THE SABOTAGE ARM REQUIRES DIVERGENCE IN `b` AND NAMES NO WRONG VALUE.
    // With one shared set, `write_set` overwrites all four bindings of that
    // set with the SECOND dispatch's (c, d, d, d) while BOTH dispatches are
    // still recorded and unsubmitted, so at submit time neither of them has
    // `b` in the output binding and `b` is written by nothing. Two outcomes
    // are therefore reachable and no third one is: `b` keeps the -1.0 the
    // fixture primed it with (the sabotage manifested), or `b` is still 1.0
    // (the driver captured the descriptors at record time, so the sabotage did
    // not manifest and the test says UNMEASURED).
    //
    // An earlier version of this arm asserted `sab_b == twos` -- b holding the
    // second dispatch's source value. That is unreachable on any driver: it
    // would need a TORN set with `c` at binding 0 and `b` still at binding 3,
    // and `write_set` updates the four bindings in one grouped call. The arm
    // therefore had no passing success path, and a sabotage that worked
    // exactly as designed would have gone red with a message denying it. It is
    // now shaped like the barrier falsifier below: require divergence from the
    // correct arm's own answer, report what was observed, and assert nothing
    // about the particular value the divergence takes -- rewriting a
    // descriptor set a recorded command buffer already names is undefined by
    // the Vulkan specification, and this test makes no claim about how any
    // driver resolves it.
    let Some(ctx) = open_or_unmeasured("descriptor arena falsifier") else {
        return;
    };
    let n = 1024u64;
    let a = ctx.buffer(n * 4).expect("a");
    let b = ctx.buffer(n * 4).expect("b");
    let c = ctx.buffer(n * 4).expect("c");
    let d = ctx.buffer(n * 4).expect("d");
    let ones = vec![1.0f32; n as usize];
    let twos = vec![2.0f32; n as usize];
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let view = contiguous(&shape);
    let (plan_ab, push_ab) = copy_plan(&shape, &a, view, &b);
    let (plan_cd, push_cd) = copy_plan(&shape, &c, view, &d);
    let groups = plan_ab.dispatch_groups();
    assert_eq!(groups, plan_cd.dispatch_groups());

    let run = |ctx: &Context| -> (Vec<f32>, Vec<f32>) {
        ctx.upload(&a, &f32_bytes(&ones)).expect("a = 1");
        ctx.upload(&c, &f32_bytes(&twos)).expect("c = 2");
        ctx.upload(&b, &f32_bytes(&vec![-1.0f32; n as usize]))
            .expect("b = -1");
        ctx.upload(&d, &f32_bytes(&vec![-1.0f32; n as usize]))
            .expect("d = -1");
        ctx.bind(&kernel, &[&a, &b, &b, &b]).expect("bind a -> b");
        ctx.dispatch(&kernel, &push_ab, groups, 1, &access_for(&a, &b))
            .expect("record a -> b");
        ctx.bind(&kernel, &[&c, &d, &d, &d]).expect("bind c -> d");
        ctx.dispatch(&kernel, &push_cd, groups, 1, &access_for(&c, &d))
            .expect("record c -> d");
        let got_b = from_bytes(&ctx.download(&b).expect("download b"));
        let got_d = from_bytes(&ctx.download(&d).expect("download d"));
        (got_b, got_d)
    };

    let (got_b, got_d) = run(&ctx);
    assert_eq!(
        got_b, ones,
        "the first dispatch wrote b from a (this assertion is what goes red if the arena is removed)"
    );
    assert_eq!(got_d, twos, "the second dispatch wrote d from c");

    #[cfg(feature = "batch-sabotage")]
    {
        use alelyon_compute_kit::DescriptorMode;
        ctx.set_descriptor_mode(DescriptorMode::SingleSet);
        let (sab_b, sab_d) = run(&ctx);
        ctx.set_descriptor_mode(DescriptorMode::Arena);
        // the conjunct on d matters: a card that leaves b right and d wrong is
        // an outcome neither arm predicts, and it must be reported rather than
        // written off as "the sabotage did not manifest"
        if sab_b == got_b && sab_d == got_d {
            eprintln!(
                "UNMEASURED here (descriptor arena falsifier): reverting to one descriptor set \
                 per kernel did NOT change either answer on this card, so the arena is a \
                 correct-by-construction claim with a control and no falsifier. This is not \
                 evidence that the arena is unnecessary and it is not evidence that it is \
                 load-bearing."
            );
        } else {
            // The divergence IS the observation. Report the value rather than
            // asserting it: which wrong answer a card produces here is
            // undefined behaviour, and pinning one would make this test a
            // claim about a driver.
            eprintln!(
                "observed: reverting to one descriptor set per kernel changed the answer -- \
                 b[0] = {} (arena: {}), d[0] = {} (arena: {}); b diverged: {}, d diverged: {}",
                sab_b[0],
                got_b[0],
                sab_d[0],
                got_d[0],
                sab_b != got_b,
                sab_d != got_d
            );
        }
    }
    #[cfg(not(feature = "batch-sabotage"))]
    eprintln!(
        "UNMEASURED here (descriptor arena falsifier): the control passed; the sabotage arm \
         needs the `batch-sabotage` feature"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(a);
    ctx.destroy_buffer(b);
    ctx.destroy_buffer(c);
    ctx.destroy_buffer(d);
}

#[test]
fn a_dependent_chain_in_one_batch_is_ordered_by_the_barrier() {
    // R1's falsifier. Dispatch A copies S into X with the identity mapping, so
    // workgroup w of A writes the elements [w*E, (w+1)*E). Dispatch B copies X
    // into Y through a TRANSPOSED view, so B's workgroup 0 reads
    // X[0], X[C], X[2C], ... -- elements A's LATE workgroups write. X is
    // primed with a sentinel first, so a read that arrives before A's write
    // shows the sentinel instead of being right by luck.
    //
    // Whether the driver in front of this test overlaps them at all is
    // UNMEASURED, and the test says so rather than claiming the barrier was
    // proved load-bearing.
    let Some(ctx) = open_or_unmeasured("barrier falsifier") else {
        return;
    };
    let rows = 512u64;
    let cols = 512u64;
    let n = rows * cols;
    assert!(
        cols >= u64::from(ELEMENTS_PER_GROUP) / 4,
        "the transposed read must span several of the writer's workgroups"
    );
    let s = ctx.buffer(n * 4).expect("s");
    let x = ctx.buffer(n * 4).expect("x");
    let y = ctx.buffer(n * 4).expect("y");
    let values: Vec<f32> = (0..n).map(|i| (i % 65_536) as f32 + 1.0).collect();
    let sentinel = vec![-7.0f32; n as usize];
    ctx.upload(&s, &f32_bytes(&values)).expect("s");
    let kernel = pointwise_kernel(&ctx);

    let a_shape = [rows, cols];
    let (a_plan, a_push) = copy_plan(&a_shape, &s, contiguous(&a_shape), &x);
    let b_shape = [cols, rows];
    let transposed = StridedView::new(0, [1, cols, 0, 0]);
    let (b_plan, b_push) = copy_plan(&b_shape, &x, transposed, &y);

    let mut reference = vec![0.0f32; n as usize];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            reference[c * rows as usize + r] = values[r * cols as usize + c];
        }
    }

    let round = |ctx: &Context| -> Vec<f32> {
        ctx.upload(&x, &f32_bytes(&sentinel)).expect("prime x");
        ctx.upload(&y, &f32_bytes(&sentinel)).expect("prime y");
        ctx.bind(&kernel, &[&s, &x, &x, &x]).expect("bind s -> x");
        ctx.dispatch(
            &kernel,
            &a_push,
            a_plan.dispatch_groups(),
            1,
            &access_for(&s, &x),
        )
        .expect("record s -> x");
        ctx.bind(&kernel, &[&x, &y, &y, &y]).expect("bind x -> y");
        ctx.dispatch(
            &kernel,
            &b_push,
            b_plan.dispatch_groups(),
            1,
            &access_for(&x, &y),
        )
        .expect("record x -> y");
        assert_eq!(ctx.recorded_items(), 2, "one batch, not two");
        from_bytes(&ctx.download(&y).expect("download y"))
    };

    const REPS: usize = 50;
    for rep in 0..REPS {
        assert_eq!(
            round(&ctx),
            reference,
            "rep {rep}: the barriered arm must be exact"
        );
    }

    #[cfg(feature = "batch-sabotage")]
    {
        use alelyon_compute_kit::BarrierMode;
        ctx.set_barrier_mode(BarrierMode::None);
        let mut diverged = 0usize;
        for _ in 0..REPS {
            if round(&ctx) != reference {
                diverged += 1;
            }
        }
        ctx.set_barrier_mode(BarrierMode::All);
        if diverged == 0 {
            eprintln!(
                "UNMEASURED here (barrier falsifier): removing every barrier from the batch did \
                 NOT diverge in {REPS} repetitions on this card, so R1 is a correct-by-construction \
                 claim with a control and no falsifier. This is not evidence that the barrier is \
                 unnecessary and it is not evidence that it is load-bearing."
            );
        } else {
            eprintln!("observed: the un-barriered arm diverged in {diverged}/{REPS} repetitions");
        }
    }
    #[cfg(not(feature = "batch-sabotage"))]
    eprintln!(
        "UNMEASURED here (barrier falsifier): the control passed; the sabotage arm needs the \
         `batch-sabotage` feature"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(s);
    ctx.destroy_buffer(x);
    ctx.destroy_buffer(y);
}

#[test]
fn a_timed_dispatch_still_returns_its_own_duration_after_flushing_the_batch() {
    let Some(ctx) = open_or_unmeasured("timed dispatch") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let out = ctx.buffer(n * 2).expect("bf16 out");
    ctx.upload(&src, &f32_bytes(&vec![1.5f32; n as usize]))
        .expect("upload");
    let cast = ctx.kernel(CAST_F32_BF16, 2, 4).expect("cast kernel");
    ctx.bind(&cast, &[&src, &out]).expect("bind");
    let count = n as u32;
    let ms = ctx
        .dispatch_timed(
            &cast,
            &count.to_le_bytes(),
            [(n as u32).div_ceil(1024), 1, 1],
            1,
        )
        .expect("timed dispatch");
    assert!(ms.is_finite() && ms >= 0.0, "a real device duration: {ms}");
    // and it is repeatable with nothing recorded in between
    let again = ctx
        .dispatch_timed(
            &cast,
            &count.to_le_bytes(),
            [(n as u32).div_ceil(1024), 1, 1],
            1,
        )
        .expect("timed dispatch again");
    assert!(again.is_finite() && again >= 0.0);
    assert_eq!(
        ctx.descriptor_watermark(&cast),
        1,
        "each timed dispatch flushes first, so the arena does not grow without bound"
    );
    ctx.destroy_kernel(cast);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(out);
}

#[test]
fn a_dispatch_declaring_a_freed_buffer_is_still_refused_freed_before_any_recording() {
    let Some(ctx) = open_or_unmeasured("freed after bind") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![1.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect("a first recorded dispatch");
    let src_id = src.id();
    ctx.destroy_buffer(src);
    let err = ctx
        .dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect_err("a dispatch on a freed buffer must never reach the queue");
    assert!(matches!(err, AckError::Freed(_)), "{err}");
    assert!(format!("{err}").contains(&format!("{src_id}")), "{err}");
    assert_eq!(
        ctx.recorded_items(),
        1,
        "the refusal added nothing to the batch"
    );
    ctx.sync().expect("drain");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(dst);
}

#[test]
fn retiring_past_the_budget_flushes_the_batch() {
    let Some(ctx) = open_or_unmeasured("retirement budget") else {
        return;
    };
    // THE UN-POOLED ARM (see the retirement test above for why). The budget
    // itself is unaffected by the pool -- `free_buffer` still retires and
    // still calls `BatchPlan::retire`, because the pool takes the pair at
    // `release_retired` and not at the free -- but the object delta across the
    // flush is 2 with the pool on rather than 4. The pooled counterpart is
    // `the_retirement_budget_still_flushes_with_the_pool_on` in
    // tests/buffer_pool.rs.
    ctx.set_buffer_pool_enabled(false);
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![4.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    // a budget smaller than one temporary, so the first retirement crosses it
    ctx.set_max_retired_bytes(n * 4 / 2);
    let doomed = ctx.buffer(n * 4).expect("doomed");
    // sampled before the first recorded item, so the batch's own command
    // buffer and fence are inside the delta (see the retirement test above)
    let before_batch = ctx.live_objects();
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    let objects = ctx.live_objects();
    assert_eq!(
        objects,
        before_batch + 2,
        "the first recorded item allocates the batch's command buffer and fence"
    );
    let before = ctx.flushes_by_reason();
    let torn_before = ctx.submissions_torn_down();
    ctx.destroy_buffer(doomed);
    only(
        before,
        ctx.flushes_by_reason(),
        FlushReason::RetiredBytes,
        1,
    );
    assert_eq!(ctx.recorded_items(), 0, "the budget flushed the batch");
    assert_eq!(
        ctx.live_objects(),
        objects - 4,
        "across the flush: what the free retired (2), and the batch's own fence and command buffer (2)"
    );
    assert_eq!(
        ctx.live_objects(),
        before_batch - 2,
        "net: only the freed buffer's pair is gone"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 1,
        "one submission released per flush"
    );
    ctx.set_max_retired_bytes(alelyon_compute_kit::batch::MAX_RETIRED_BYTES);
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

// ---------------------------------------------------------------------------
// The flush's ACK-L0-03 ladder, and the one flush point the counter cannot see
//
// The arms below need the `fault-injection` feature AND a device. Without the
// feature they are absent from the build; without a device they print
// UNMEASURED and return. What they drive is the ladder `Context::flush`
// repeats from `one_shot` with "the call's objects" widened to "the batch's":
// before this file existed, every armed fault in the suite was consumed by
// `one_shot`, because `upload`, `download` and `dispatch_timed` -- the only
// methods `tests/poison.rs` and `tests/leaks.rs` drive -- flush an EMPTY
// recorder and return before any Vulkan call. So the flush's own ladder had no
// test at all, and its leak arithmetic is a second, hand-copied site whose
// twin in `one_shot` was the only one under test.
//
// STILL UNMEASURED after these: the three pre-submission arms
// (`end_command_buffer`, `reset_fences`, `queue_submit`). `Fault` can
// substitute a fence-wait result, a device-idle result and a creation result,
// and nothing else, so those three need a real driver fault.
// ---------------------------------------------------------------------------

/// Record one dispatch of a copy into an open batch, and hand back the pieces
/// the caller still owns plus `live_objects()` sampled IMMEDIATELY BEFORE the
/// dispatch -- so a caller's delta covers the batch's own command buffer and
/// fence rather than smuggling them into its baseline. The batch is open when
/// this returns.
#[cfg(feature = "fault-injection")]
fn record_one(ctx: &Context) -> (Buffer, Buffer, Kernel, i64) {
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![1.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    let before_batch = ctx.live_objects();
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    assert_eq!(ctx.recorded_items(), 1, "a batch is open");
    (src, dst, kernel, before_batch)
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_flush_whose_wait_fails_on_a_device_that_quiesces_releases_the_batch_and_stays_usable() {
    use alelyon_compute_kit::Fault;
    let Some(ctx) = open_or_unmeasured("flush ladder: quiesced") else {
        return;
    };
    let (src, dst, kernel, before_batch) = record_one(&ctx);
    let open = ctx.live_objects();
    assert_eq!(
        open,
        before_batch + 2,
        "the first recorded item allocates the batch's command buffer and fence"
    );
    let torn_before = ctx.submissions_torn_down();
    let leaked_before = ctx.leaked_on_poison();

    ctx.inject_fault(Fault::WaitFails(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx
        .sync()
        .expect_err("the injected wait failure must surface");
    assert!(
        matches!(
            err,
            AckError::Vulkan(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ),
        "the wait's own result reaches the caller: {err}"
    );
    assert!(
        !ctx.is_poisoned(),
        "the device quiesced, so the batch ran and the context stays usable"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 1,
        "quiesced: the batch's fence and command buffer are released, once for the flush"
    );
    assert_eq!(
        ctx.leaked_on_poison(),
        leaked_before,
        "released, not leaked"
    );
    assert_eq!(
        ctx.live_objects(),
        open - 2,
        "the batch's own pair is gone; the caller's buffers are not"
    );
    assert_eq!(ctx.recorded_items(), 0, "the batch is closed either way");

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_flush_that_cannot_quiesce_leaks_the_batch_and_everything_retired_into_it() {
    use alelyon_compute_kit::Fault;
    let Some(ctx) = open_or_unmeasured("flush ladder: unquiesced") else {
        return;
    };
    let (src, dst, kernel, before_batch) = record_one(&ctx);
    // one buffer freed into the open batch, so the leak count has to cover the
    // retirement list and not just the submission's own two objects
    let doomed = ctx.buffer(4096).expect("doomed");
    ctx.destroy_buffer(doomed);
    let open = ctx.live_objects();
    assert_eq!(
        open,
        before_batch + 4,
        "the batch's pair, and the retired buffer's pair still counted"
    );
    let torn_before = ctx.submissions_torn_down();
    let leaked_before = ctx.leaked_on_poison();

    let oom = ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    ctx.inject_fault(Fault::WaitFails(oom));
    ctx.inject_fault(Fault::IdleFails(oom));
    let err = ctx.sync().expect_err("an unquiesced device must poison");
    assert!(matches!(err, AckError::Poisoned(_)), "{err}");
    let text = format!("{err}");
    assert!(
        text.contains("1 recorded dispatch"),
        "the error names the BATCH, not one operation: {text}"
    );
    assert!(ctx.is_poisoned());
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before,
        "nothing was released, so nothing may be counted as released"
    );
    assert_eq!(
        ctx.leaked_on_poison(),
        leaked_before + 4,
        "the fence and the command buffer (2), and the retired buffer and its memory (2)"
    );
    assert_eq!(
        ctx.live_objects(),
        open,
        "and live_objects agrees: leaked objects stay counted"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_flush_that_reports_a_lost_device_releases_the_batch_and_leaks_nothing() {
    use alelyon_compute_kit::Fault;
    let Some(ctx) = open_or_unmeasured("flush ladder: device lost") else {
        return;
    };
    let (src, dst, kernel, before_batch) = record_one(&ctx);
    let open = ctx.live_objects();
    assert_eq!(open, before_batch + 2, "the batch's own pair");
    let torn_before = ctx.submissions_torn_down();
    let leaked_before = ctx.leaked_on_poison();

    ctx.inject_fault(Fault::WaitFails(ash::vk::Result::ERROR_DEVICE_LOST));
    let err = ctx.sync().expect_err("a lost device must surface");
    assert!(
        matches!(err, AckError::Vulkan(ash::vk::Result::ERROR_DEVICE_LOST)),
        "an established loss reports the real VkResult, not prose: {err}"
    );
    assert!(ctx.is_poisoned());
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_before + 1,
        "a lost device permits teardown"
    );
    assert_eq!(
        ctx.leaked_on_poison(),
        leaked_before,
        "a lost device permits teardown, so nothing is leaked to count"
    );
    assert_eq!(
        ctx.live_objects(),
        open - 2,
        "the batch's pair was released"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_batch_that_cannot_open_its_fence_frees_the_command_buffer_it_had_created() {
    // ACK-L0-05 for `open_batch`, which is the one creation site this
    // increment added: a failure at the fence must not strand the command
    // buffer allocated a line above it, and must install no recorder.
    use alelyon_compute_kit::{CreateStep, Fault};
    let Some(ctx) = open_or_unmeasured("open_batch cleanup") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![1.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    let access = access_for(&src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");

    let before = ctx.live_objects();
    ctx.inject_fault(Fault::CreateFails(
        CreateStep::Fence,
        ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    let err = ctx
        .dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect_err("the batch could not be opened");
    assert!(
        matches!(
            err,
            AckError::Vulkan(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ),
        "{err}"
    );
    assert_eq!(
        ctx.live_objects(),
        before,
        "the command buffer allocated before the fence must be freed again"
    );
    assert_eq!(ctx.recorded_items(), 0, "no recorder was installed");
    assert!(!ctx.is_poisoned(), "a creation failure is not a poison");

    // and the next dispatch opens a batch normally
    ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect("a batch opens after the failure");
    assert_eq!(ctx.recorded_items(), 1);
    ctx.sync().expect("drain");

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_close_with_work_recorded_flushes_and_reports_the_failure() {
    // `FlushReason::Close`'s only producer is `ffi::ack_close`, and neither the
    // Rust API nor the C ABI can be asked for that flush or read its counter
    // (`Context::flush` is `pub(crate)`, `sync` hardcodes `ExplicitSync`, and
    // no C entry exposes `flushes_by_reason`). So the branch is pinned through
    // the one thing it does emit: the message a FAILED close flush leaves in
    // `ack_last_error`, which names the reason and the batch's item count. A
    // deleted close flush cannot produce it, and neither can any other flush
    // point -- every other one returns its error to its own caller.
    //
    // UNMEASURED: the SUCCESS path of the same flush. At ABI 9 a successful
    // close writes nothing, the device is gone before anything can be read
    // back, and `ack_close` returns `ACK_OK` either way, so nothing
    // distinguishes "flushed" from "silently discarded" from outside.
    use alelyon_compute_kit::ffi;
    use std::ffi::{c_char, CStr};

    fn last_error() -> String {
        let mut buf = vec![0 as c_char; 2048];
        let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
        assert!(rc >= 0, "ack_last_error returned {rc}");
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    // nothing recorded: the close makes no flush and writes no message.
    // One device at a time, so this is also the UNMEASURED gate.
    let quiet = ffi::ack_open();
    if quiet.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "close flush: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (close flush): {}", last_error());
        return;
    }
    let before_quiet = last_error();
    assert_eq!(ffi::ack_close(quiet), ffi::ACK_OK, "{}", last_error());
    assert_eq!(
        last_error(),
        before_quiet,
        "an empty batch is not flushed at close, so nothing is written over the last message"
    );

    let dev = ffi::ack_open();
    assert!(!dev.is_null(), "{}", last_error());
    let n = 1024u64;
    let src = ffi::ack_buffer_alloc(dev, n * 4);
    let dst = ffi::ack_buffer_alloc(dev, n * 2);
    assert!(!src.is_null() && !dst.is_null(), "{}", last_error());
    let data = f32_bytes(&vec![1.5f32; n as usize]);
    assert_eq!(
        unsafe { ffi::ack_upload(dev, src, data.as_ptr(), data.len()) },
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    // ack_cast RECORDS; nothing has submitted it when the frees run
    assert_eq!(
        ffi::ack_cast(dev, src, dst, n, 0),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    // close refuses while a buffer is live, and a free with a batch open
    // retires rather than destroys, so the batch is still non-empty here
    assert_eq!(
        ffi::ack_buffer_free(dev, src),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(
        ffi::ack_buffer_free(dev, dst),
        ffi::ACK_OK,
        "{}",
        last_error()
    );

    let oom = ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw();
    assert_eq!(ffi::ack_inject_fault(dev, 0, oom), ffi::ACK_OK);
    assert_eq!(ffi::ack_inject_fault(dev, 1, oom), ffi::ACK_OK);
    assert_eq!(
        ffi::ack_close(dev),
        ffi::ACK_OK,
        "a flush failure does not change this entry's code (ABI 9)"
    );
    let msg = last_error();
    assert!(
        msg.contains("close flushed the deferred batch and it failed"),
        "the close's own flush must be reported, not swallowed: {msg}"
    );
    assert!(
        msg.contains("reason Close"),
        "and the message must name the reason, which only this flush point can: {msg}"
    );
    assert!(
        msg.contains("1 recorded dispatch"),
        "one recorded dispatch was in the batch the close submitted: {msg}"
    );
}

#[test]
fn a_kernel_destroyed_while_a_batch_is_open_is_retired_and_the_flush_destroys_it() {
    // The kernel's half of retirement. A recorded, unsubmitted dispatch holds
    // `cmd_bind_pipeline` on this kernel's pipeline and
    // `cmd_bind_descriptor_sets` on a set from its pool, so destroying either
    // before the submission moves the command buffer to the invalid state and
    // the flush's `vkQueueSubmit` is then undefined -- not an error this crate
    // could report. Before deferred submission the same three lines were safe
    // because every dispatch was submitted and waited for before `dispatch`
    // returned; they are not any more.
    //
    // The correctness assertion is the load-bearing one: the recorded dispatch
    // must still produce the right bytes at the flush, which it can only do if
    // the pipeline and the descriptor set it names are still alive when the
    // submission happens.
    let Some(ctx) = open_or_unmeasured("kernel retirement") else {
        return;
    };
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");

    let open = ctx.live_objects();
    assert_eq!(ctx.recorded_items(), 1, "a batch is open");
    ctx.destroy_kernel(kernel);
    assert_eq!(
        ctx.live_objects(),
        open,
        "recorded commands still name this pipeline and a set from its pool: the four objects are retired, not destroyed, and stay counted"
    );

    ctx.sync().expect("flush");
    assert_eq!(
        ctx.live_objects(),
        open - 6,
        "across the flush: the kernel's four objects, and the batch's own fence and command buffer"
    );
    assert_eq!(
        from_bytes(&ctx.download(&dst).expect("download")),
        values,
        "the recorded dispatch ran against a live pipeline and set"
    );

    // with no batch open the destroy is immediate, as it always was
    let again = pointwise_kernel(&ctx);
    let before = ctx.live_objects();
    assert_eq!(ctx.recorded_items(), 0, "no batch is open");
    ctx.destroy_kernel(again);
    assert_eq!(
        ctx.live_objects(),
        before - 4,
        "destroyed, not retired: the pool, pipeline, pipeline layout and set layout"
    );

    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

// ---------------------------------------------------------------------------
// Asynchronous flushes (2026-09-10): a `BatchFull` flush submits its batch and
// leaves it in flight; the next flush finishes it; a flush that promises
// completion (a readback here) finishes it at once. With the timestamps
// instrument on every flush completes, so these run only with it off.

fn timestamps_off() -> bool {
    match std::env::var_os("ACK_DISPATCH_TIMESTAMPS") {
        None => true,
        Some(v) => v == "0",
    }
}

/// A chain of `links` copies, each reading the previous link's output, with
/// the batch depth `depth`: the values must arrive at the end whatever the
/// batches were, which is the ordering law across submissions (a barrier in
/// a later batch covers every command earlier in submission order).
fn copy_chain(
    ctx: &Context,
    depth: u32,
    links: usize,
) -> (Vec<f32>, Vec<f32>, Vec<Buffer>, Kernel) {
    ctx.set_max_recorded_items(depth);
    let n = 2048u64;
    let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let mut buffers = Vec::with_capacity(links + 1);
    for _ in 0..=links {
        buffers.push(ctx.buffer(n * 4).expect("link"));
    }
    ctx.upload(&buffers[0], &f32_bytes(&values))
        .expect("upload");
    let kernel = pointwise_kernel(ctx);
    let shape = [n];
    for i in 0..links {
        let (src, dst) = (&buffers[i], &buffers[i + 1]);
        let (plan, push) = copy_plan(&shape, src, contiguous(&shape), dst);
        ctx.bind(&kernel, &[src, dst, dst, dst]).expect("bind");
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
    (values, back, buffers, kernel)
}

#[test]
fn a_full_batch_is_left_in_flight_and_the_readback_finishes_it() {
    let Some(ctx) = open_or_unmeasured("batch in flight") else {
        return;
    };
    if !timestamps_off() {
        eprintln!("UNMEASURED here: the timestamps instrument completes every flush");
        return;
    }
    ctx.set_max_recorded_items(2);
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let mid = ctx.buffer(n * 4).expect("mid");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    assert_eq!(ctx.batches_in_flight(), 0, "nothing submitted yet");
    let before = ctx.flushes_by_reason();
    for (a, b) in [(&src, &mid), (&mid, &dst)] {
        let (plan, push) = copy_plan(&shape, a, contiguous(&shape), b);
        ctx.bind(&kernel, &[a, b, b, b]).expect("bind");
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access_for(a, b))
            .expect("record");
    }
    // the second item filled the batch: submitted, not waited for
    only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 1);
    assert_eq!(ctx.recorded_items(), 0, "the batch was taken by the flush");
    assert_eq!(ctx.batches_in_flight(), 1, "and left in flight");
    // a readback promises completion: the batch in flight is finished first.
    // Nothing is recorded, so the readback's own flush is of an empty batch,
    // which the counter has never counted (see `flushes`)
    let before = ctx.flushes_by_reason();
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    only(before, ctx.flushes_by_reason(), FlushReason::Readback, 0);
    assert_eq!(ctx.batches_in_flight(), 0, "finished by the readback");
    assert_eq!(back, values, "both copies ran, in order");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(mid);
    ctx.destroy_buffer(dst);
}

#[test]
fn a_dependent_chain_across_asynchronous_batches_arrives_intact() {
    let Some(ctx) = open_or_unmeasured("chain across batches") else {
        return;
    };
    if !timestamps_off() {
        eprintln!("UNMEASURED here: the timestamps instrument completes every flush");
        return;
    }
    // depth 1: every link is its own batch, each submitted while the previous
    // one may still be running; depth 3: links straddle batches
    for depth in [1u32, 3] {
        let (values, back, buffers, kernel) = copy_chain(&ctx, depth, 8);
        assert_eq!(back, values, "depth {depth}: the chain's end is its start");
        assert_eq!(
            ctx.batches_in_flight(),
            0,
            "the readback finished the last batch"
        );
        ctx.destroy_kernel(kernel);
        for b in buffers {
            ctx.destroy_buffer(b);
        }
    }
}

#[test]
fn a_buffer_freed_with_no_batch_open_retires_into_the_batch_in_flight() {
    let Some(ctx) = open_or_unmeasured("retire into the batch in flight") else {
        return;
    };
    if !timestamps_off() {
        eprintln!("UNMEASURED here: the timestamps instrument completes every flush");
        return;
    }
    ctx.set_max_recorded_items(1);
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| 3.0 - i as f32).collect();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    // depth 1: the item's own flush submitted it and left it in flight
    assert_eq!(ctx.batches_in_flight(), 1);
    assert_eq!(ctx.recorded_items(), 0);
    // freeing the source now, with no batch open, must not hand its pair to
    // the pool while the batch in flight may still be reading it
    let live = ctx.live_objects();
    ctx.destroy_buffer(src);
    assert_eq!(
        ctx.live_objects(),
        live,
        "retired, not destroyed: the batch in flight names it"
    );
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    assert_eq!(back, values, "the copy read the source before it went");
    assert_eq!(ctx.batches_in_flight(), 0);
    // the finish released the pair: destroyed, or held by the pool (still a
    // live object either way for the pool's own accounting)
    assert!(
        ctx.live_objects() == live - 2 || ctx.pool_stats().held_bytes >= n * 4,
        "the finish released the retired pair"
    );
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(dst);
}

#[test]
fn after_a_completing_flush_the_batches_ramp_from_sixteen_to_the_depth() {
    let Some(ctx) = open_or_unmeasured("batch ramp") else {
        return;
    };
    if !timestamps_off() {
        eprintln!("UNMEASURED here: the timestamps instrument completes every flush");
        return;
    }
    use alelyon_compute_kit::batch::{MAX_RECORDED_ITEMS, RAMP_FIRST};
    ctx.set_max_recorded_items(MAX_RECORDED_ITEMS);
    let n = 256u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    // the upload is a completing flush: the ramp starts here
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    let access = access_for(&src, &dst);
    // the same copy recorded again and again: only the batch boundaries matter
    let mut expected = RAMP_FIRST;
    let mut filled = 0u64;
    while expected < MAX_RECORDED_ITEMS {
        let before = ctx.flushes_by_reason();
        for _ in 0..expected {
            ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
                .expect("record");
        }
        filled += 1;
        only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 1);
        assert_eq!(
            ctx.recorded_items(),
            0,
            "batch {filled} of {expected} items was full at exactly that count"
        );
        expected *= 2;
    }
    // at the depth: one more item than the depth is what fills the next batch
    let before = ctx.flushes_by_reason();
    for _ in 0..MAX_RECORDED_ITEMS {
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
            .expect("record");
    }
    only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 1);
    assert_eq!(ctx.recorded_items(), 0, "the ramp reached the depth");
    // a readback completes everything and resets the ramp: the next batch is short again
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    assert_eq!(back, values);
    let before = ctx.flushes_by_reason();
    for _ in 0..RAMP_FIRST {
        ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
            .expect("record");
    }
    only(before, ctx.flushes_by_reason(), FlushReason::BatchFull, 1);
    assert_eq!(
        ctx.recorded_items(),
        0,
        "after the readback the first batch is RAMP_FIRST items again"
    );
    ctx.download(&dst).expect("drain");
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn an_upload_is_recorded_into_the_batch_and_flushes_nothing() {
    let Some(ctx) = open_or_unmeasured("recorded upload") else {
        return;
    };
    if !timestamps_off() {
        eprintln!(
            "UNMEASURED here: the timestamps instrument keeps every upload its own submission"
        );
        return;
    }
    // opt-in, as a caller does with ACK_DEFERRED_UPLOAD=1 at open
    ctx.set_deferred_upload(true);
    assert!(
        ctx.deferred_upload(),
        "on: the asynchronous flush is on and the instrument off"
    );
    let n = 1024u64;
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let kernel = pointwise_kernel(&ctx);
    let values: Vec<f32> = (0..n).map(|i| i as f32 * 1.5 - 3.0).collect();
    let live = ctx.live_objects();
    let before = ctx.flushes_by_reason();
    ctx.upload(&src, &f32_bytes(&values)).expect("upload");
    // nothing flushed, nothing waited: the copy sits in the open batch
    assert_eq!(
        before,
        ctx.flushes_by_reason(),
        "a recorded upload fires no flush"
    );
    assert_eq!(
        ctx.recorded_items(),
        0,
        "the copy is not an item of the plan"
    );
    assert_eq!(
        ctx.live_objects(),
        live + 4,
        "no batch was open: the upload opened one (two handles) and retired its staging pair into it"
    );
    // the dispatch that reads the destination follows it in the same batch,
    // behind a barrier, and the readback carries them both
    let shape = [n];
    let (plan, push) = copy_plan(&shape, &src, contiguous(&shape), &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    assert_eq!(ctx.recorded_items(), 1);
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    assert_eq!(
        back, values,
        "the copy ran before the dispatch that read it"
    );
    assert_eq!(ctx.batches_in_flight(), 0);
    assert_eq!(
        ctx.live_objects(),
        live,
        "the staging pair and the batch's handles went with its finish"
    );
    // an upload into a buffer a recorded dispatch READS lands after that
    // dispatch: the barrier orders the transfer behind the compute
    let second: Vec<f32> = values.iter().map(|v| v * 2.0).collect();
    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    ctx.upload(&src, &f32_bytes(&second))
        .expect("upload after a read");
    let back = from_bytes(&ctx.download(&dst).expect("download"));
    assert_eq!(
        back, values,
        "the dispatch read the source before the later upload overwrote it"
    );
    assert_eq!(from_bytes(&ctx.download(&src).expect("source")), second);
    ctx.set_deferred_upload(false);
    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

// ---------------------------------------------------------------------------
// The declared-access barrier rule (2026-09-10): a barrier separates two items
// of a batch only when their declared accesses conflict. The test above, eight
// copies of one source into one destination, is the conflicting case and still
// reads barriers == items - 1; these are the other cases.

#[test]
fn independent_dispatches_in_one_batch_take_no_barrier_between_them() {
    let Some(ctx) = open_or_unmeasured("independent dispatches") else {
        return;
    };
    let n = 1024u64;
    let shape = [n];
    let kernel = pointwise_kernel(&ctx);
    // four disjoint pairs: nothing any of them writes is touched by another
    let mut pairs = Vec::new();
    for i in 0..4u64 {
        let src = ctx.buffer(n * 4).expect("src");
        let dst = ctx.buffer(n * 4).expect("dst");
        let values: Vec<f32> = (0..n).map(|v| v as f32 + i as f32 * 1000.0).collect();
        ctx.upload(&src, &f32_bytes(&values)).expect("upload");
        pairs.push((src, dst, values));
    }
    let before = ctx.recorded_barriers();
    assert_eq!(before, 0, "the uploads left no batch open");
    for (src, dst, _) in &pairs {
        let (plan, push) = copy_plan(&shape, src, contiguous(&shape), dst);
        ctx.bind(&kernel, &[src, dst, dst, dst]).expect("bind");
        ctx.dispatch(
            &kernel,
            &push,
            plan.dispatch_groups(),
            1,
            &access_for(src, dst),
        )
        .expect("record");
    }
    assert_eq!(ctx.recorded_items(), 4);
    assert_eq!(
        ctx.recorded_barriers(),
        0,
        "four disjoint copies conflict with nothing, so nothing separates them"
    );
    // and the values still arrive: independence is what makes dropping it safe
    for (_, dst, values) in &pairs {
        assert_eq!(&from_bytes(&ctx.download(dst).expect("download")), values);
    }
    ctx.destroy_kernel(kernel);
    for (src, dst, _) in pairs {
        ctx.destroy_buffer(src);
        ctx.destroy_buffer(dst);
    }
}

#[test]
fn a_dependent_pair_still_takes_its_barrier_and_arrives_in_order() {
    let Some(ctx) = open_or_unmeasured("dependent pair") else {
        return;
    };
    let n = 1024u64;
    let shape = [n];
    let kernel = pointwise_kernel(&ctx);
    let a = ctx.buffer(n * 4).expect("a");
    let b = ctx.buffer(n * 4).expect("b");
    let c = ctx.buffer(n * 4).expect("c");
    let far = ctx.buffer(n * 4).expect("far");
    let far_out = ctx.buffer(n * 4).expect("far out");
    let values: Vec<f32> = (0..n).map(|v| v as f32 * 0.5 - 3.0).collect();
    ctx.upload(&a, &f32_bytes(&values)).expect("upload a");
    ctx.upload(&far, &f32_bytes(&values)).expect("upload far");
    let before = ctx.recorded_barriers();
    // a -> b, then an INDEPENDENT far -> far_out, then b -> c which reads what
    // the first wrote: one barrier, before the third item and not before the second
    for (src, dst) in [(&a, &b), (&far, &far_out), (&b, &c)] {
        let (plan, push) = copy_plan(&shape, src, contiguous(&shape), dst);
        ctx.bind(&kernel, &[src, dst, dst, dst]).expect("bind");
        ctx.dispatch(
            &kernel,
            &push,
            plan.dispatch_groups(),
            1,
            &access_for(src, dst),
        )
        .expect("record");
    }
    assert_eq!(ctx.recorded_items(), 3);
    assert_eq!(
        ctx.recorded_barriers() - before,
        1,
        "one barrier: before the item that reads what an earlier one wrote, and no other"
    );
    assert_eq!(
        from_bytes(&ctx.download(&c).expect("download")),
        values,
        "the chain arrived through the barrier"
    );
    assert_eq!(
        from_bytes(&ctx.download(&far_out).expect("download far")),
        values
    );
    ctx.destroy_kernel(kernel);
    for buffer in [a, b, c, far, far_out] {
        ctx.destroy_buffer(buffer);
    }
}

#[test]
fn the_always_rule_is_reachable_and_restores_a_barrier_between_every_pair() {
    // ACK_BARRIERS is read once per BatchPlan, so this exercises the rule
    // through the plan directly rather than by setting a process-wide variable
    // while another test may be opening a context.
    use alelyon_compute_kit::batch::{Access, BarrierRule, BatchPlan};
    let mut declared = BatchPlan::new(0, 64, 1 << 30);
    let one = [Access::read(1), Access::write(2)];
    let other = [Access::read(3), Access::write(4)];
    assert!(
        !declared.record_with(&one),
        "no barrier before the first item"
    );
    assert!(!declared.record_with(&other), "disjoint: no barrier");
    assert!(
        declared.record_with(&[Access::read(2), Access::write(5)]),
        "reads what item one wrote"
    );
    assert!(
        !declared.record_with(&[Access::read(9), Access::write(8)]),
        "disjoint again"
    );
    assert!(
        declared.record_with(&[]),
        "an empty declaration is a conflict, not independence"
    );
    // two: before the item that reads what item one wrote, and before the
    // empty declaration. The pair after the first barrier is disjoint from
    // what followed it, so nothing separates them.
    assert_eq!(declared.barriers(), 2);
    assert_eq!(
        BarrierRule::from_env(),
        BarrierRule::Declared,
        "the default rule"
    );
}
