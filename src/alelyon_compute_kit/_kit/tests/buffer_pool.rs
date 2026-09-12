//! The buffer pool: what it recycles, when it is allowed to, and the one
//! wrong answer it could produce.
//!
//! `Context::buffer` is 219 us at 4 KiB and `destroy_buffer` is 193 us on this
//! host, measured at 800 live allocations, and the registered A2 microbatch
//! pass makes 3,565 of those lifetimes -- roughly six tenths of the pass. The
//! pool holds a freed buffer's `(VkBuffer, VkDeviceMemory)` pair instead of
//! destroying it, keyed by the size `raw_buffer` was called with.
//!
//! IT IS A FREE LIST AND NOT A SUBALLOCATOR, and that is a correctness
//! decision. Eight refusals in the crate read `Buffer::id` inequality as
//! memory-disjointness; two live ids on one Vulkan allocation would make all
//! eight go silent at once, in the direction of admitting the dispatch. One
//! `VkBuffer` per handle keeps "at most one live id per physical allocation"
//! true by construction.
//!
//! THE FALSIFIER IS THE LOAD-BEARING TEST HERE. `a_pooled_pair_is_not_handed
//! _back_while_a_recorded_dispatch_still_names_it` runs one fixture twice in
//! one session, differing in ONE input -- `PoolMode` -- and requires the
//! sabotaged arm to DIVERGE, by reading back exactly the bytes a pending
//! dispatch wrote into what its owner believed was a fresh allocation. If the
//! sabotage does not manifest the test reports UNMEASURED and fails rather
//! than passing: a sabotage that never applied reads as a passing row. It
//! needs `--features pool-sabotage`; without it the sabotage arm is compiled
//! out and the test says so instead of claiming the evidence.
//!
//! EVERY TEST BRACKETS THE POOL'S OWN ACCOUNTING as well as its arithmetic. A
//! correctness probe cannot tell you a buffer was recycled rather than freshly
//! allocated -- a pool that silently allocated every time would satisfy every
//! value assertion below. The hit/miss bracket is what makes them non-vacuous.
//!
//! Needs a Vulkan device. Without one every test here prints UNMEASURED and
//! returns, which is a skip with its reason, never a pass; set
//! `ACK_REQUIRE_DEVICE=1` to turn that into a failure on a machine that is
//! supposed to have the device. A GREEN RUN WITHOUT THAT VARIABLE IS NOT
//! DEVICE EVIDENCE.

use alelyon_compute_kit::pointwise_ops::{
    PointwiseOp, PointwisePlan, Scalars, Storage, StridedView, OPERANDS, PUSH_BYTES,
};
use alelyon_compute_kit::{Access, AckError, Buffer, Context, Kernel};
#[cfg(feature = "fault-injection")]
use alelyon_compute_kit::{CreateStep, Fault};
#[cfg(feature = "fault-injection")]
use ash::vk;

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

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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

fn copy_plan(shape: &[u64], src: &Buffer, dst: &Buffer) -> (PointwisePlan, [u8; PUSH_BYTES]) {
    let view = contiguous(shape);
    let views: [StridedView; OPERANDS] = [view, view, view, view];
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

fn access_for(src: &Buffer, dst: &Buffer) -> [Access; OPERANDS] {
    [
        Access::read(src.id()),
        Access::read(dst.id()),
        Access::read(dst.id()),
        Access::write(dst.id()),
    ]
}

// ---------------------------------------------------------------------------
// what the pool does
// ---------------------------------------------------------------------------

#[test]
fn a_free_with_nothing_recorded_returns_the_pair_and_the_next_request_reuses_it() {
    let Some(ctx) = open_or_unmeasured("pool hit") else {
        return;
    };
    let n = 4096u64;
    let before = ctx.pool_stats();
    let b = ctx.buffer(n).expect("first");
    let after_alloc = ctx.pool_stats();
    assert_eq!(
        after_alloc.misses,
        before.misses + 1,
        "an empty pool cannot serve the first request"
    );
    assert_eq!(after_alloc.hits, before.hits, "and it did not claim to");

    let objects_with_one_live = ctx.live_objects();
    ctx.destroy_buffer(b);
    let after_free = ctx.pool_stats();
    assert_eq!(
        after_free.returns,
        before.returns + 1,
        "no batch is open, so the pair goes straight to the pool"
    );
    assert_eq!(
        ctx.live_objects(),
        objects_with_one_live,
        "a pooled pair is LIVE, not leaked and not destroyed: live_objects() counts it"
    );
    assert_eq!(
        ctx.pooled_objects(),
        2,
        "and pooled_objects() names how much of live_objects() the pool is holding"
    );

    let c = ctx.buffer(n).expect("second");
    let after_reuse = ctx.pool_stats();
    assert_eq!(
        after_reuse.hits,
        before.hits + 1,
        "the second request of the same size is served from the pool"
    );
    assert_eq!(
        after_reuse.misses, after_alloc.misses,
        "and it did not allocate"
    );
    assert_eq!(
        ctx.live_objects(),
        objects_with_one_live,
        "a hit creates no Vulkan object"
    );
    assert_eq!(ctx.pooled_objects(), 0, "and the pool is empty again");
    ctx.destroy_buffer(c);
}

#[test]
fn a_recycled_pair_carries_a_fresh_id_so_a_stale_binding_is_still_refused() {
    // `check_dispatch` decides a bound buffer is still live by
    // `live_buffers.contains(&id)`, and `Kernel` stores ids, not handles. If a
    // pool recycled the id with the pair, a kernel bound to a dead buffer
    // would pass that check and dispatch against the NEW owner's tensor while
    // reporting success. This pins that it does not.
    let Some(ctx) = open_or_unmeasured("fresh id on a hit") else {
        return;
    };
    let n = 1024u64;
    let shape = [n];
    let src = ctx.buffer(n * 4).expect("src");
    let doomed = ctx.buffer(n * 4).expect("doomed");
    ctx.upload(&src, &f32_bytes(&vec![1.5f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    ctx.bind(&kernel, &[&src, &doomed, &doomed, &doomed])
        .expect("bind");
    let (plan, push) = copy_plan(&shape, &src, &doomed);
    let access = access_for(&src, &doomed);
    let doomed_id = doomed.id();

    ctx.destroy_buffer(doomed);
    let hits_before = ctx.pool_stats().hits;
    let reborn = ctx.buffer(n * 4).expect("reborn");
    assert_eq!(
        ctx.pool_stats().hits,
        hits_before + 1,
        "the pair came back from the pool: without this the rest of the test is vacuous"
    );
    assert_ne!(
        reborn.id(),
        doomed_id,
        "a recycled PAIR must not carry a recycled ID"
    );

    match ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access) {
        Err(AckError::Freed(_)) => {}
        other => panic!(
            "a kernel bound to a freed buffer must still be refused after the pair was recycled, \
             got {other:?}"
        ),
    }

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(reborn);
    ctx.destroy_buffer(src);
}

#[test]
fn a_recycled_pair_holds_the_previous_tenants_bytes_and_the_pool_says_so() {
    // The pool's stated contract: RECYCLED MEMORY IS NOT ZEROED. A fresh
    // `vkAllocateMemory` reads zero on this driver in practice (Vulkan does
    // not promise it), so relying on that was already latent; the pool makes
    // the residue deterministic and non-zero. This makes the sentence a tested
    // one rather than an assumption, and it is also the only evidence in this
    // file that a hit really returned the SAME memory rather than a hidden
    // fresh allocation the counter merely called a hit.
    let Some(ctx) = open_or_unmeasured("residue") else {
        return;
    };
    let n = 256u64;
    let pattern: Vec<f32> = (0..n).map(|i| 7.0 + i as f32).collect();
    let a = ctx.buffer(n * 4).expect("a");
    ctx.upload(&a, &f32_bytes(&pattern)).expect("upload");
    ctx.destroy_buffer(a);

    let hits_before = ctx.pool_stats().hits;
    let b = ctx.buffer(n * 4).expect("b");
    assert_eq!(
        ctx.pool_stats().hits,
        hits_before + 1,
        "the residue claim below is about a POOLED pair; a miss would make it meaningless"
    );
    let got = ctx.download(&b).expect("download");
    assert_eq!(
        got,
        f32_bytes(&pattern),
        "recycled memory is handed back as it was left: this is the pool's contract, and a \
         kernel that writes only part of its output now leaks the previous tensor instead of \
         zeros"
    );
    ctx.destroy_buffer(b);
}

#[test]
fn a_different_size_is_not_served_from_the_pool() {
    let Some(ctx) = open_or_unmeasured("size keying") else {
        return;
    };
    let a = ctx.buffer(4096).expect("a");
    ctx.destroy_buffer(a);
    let hits_before = ctx.pool_stats().hits;
    let b = ctx.buffer(8192).expect("b");
    assert_eq!(
        ctx.pool_stats().hits,
        hits_before,
        "the free list is keyed on the ALLOCATED size; a larger request must not take a smaller pair"
    );
    assert_eq!(
        b.bytes, 8192,
        "and Buffer::bytes stays the size the caller asked for, because the pointwise, reduce, \
         row and loss planners derive their capacity bounds from it"
    );
    ctx.destroy_buffer(b);
}

#[test]
fn the_pool_holds_its_objects_and_the_leak_law_still_closes() {
    // The pooled counterpart of `every_object_a_method_creates_is_destroyed_
    // again_and_the_count_says_so` in tests/leaks.rs, which now runs with the
    // pool off. ACK-L0-05 says nothing a method created may stay allocated
    // without the count saying so; the pool does not weaken that, it names a
    // second reason an object can be live. What must still be true, and is
    // asserted here, is that every object the pool holds is ACCOUNTED -- the
    // difference between `live_objects()` and `pooled_objects()` returns to
    // the baseline, and a drain closes the count exactly.
    let Some(ctx) = open_or_unmeasured("pooled leak law") else {
        return;
    };
    let base = ctx.live_objects();
    assert_eq!(base, 0, "a fresh context counts nothing");
    let a = ctx.buffer(4096).expect("a");
    let b = ctx.buffer(8192).expect("b");
    assert_eq!(
        ctx.live_objects(),
        base + 4,
        "two buffers, two objects each"
    );
    ctx.destroy_buffer(a);
    ctx.destroy_buffer(b);
    assert_eq!(
        ctx.live_objects(),
        base + 4,
        "the pool is holding both pairs, and they are live objects"
    );
    assert_eq!(
        ctx.pooled_objects(),
        4,
        "and it says exactly how many of them are its"
    );
    assert_eq!(
        ctx.live_objects() - ctx.pooled_objects(),
        base,
        "back to the baseline, once what the pool holds is named"
    );
    assert_eq!(
        ctx.leaked_on_poison(),
        0,
        "a healthy context leaks nothing: HELD is not LEAKED"
    );
    ctx.set_buffer_pool_enabled(false);
    assert_eq!(
        ctx.live_objects(),
        base,
        "and the drain closes the count exactly, with no object unaccounted for"
    );
    assert_eq!(
        ctx.pool_stats().drained,
        2,
        "two pairs destroyed by the drain"
    );
}

#[test]
#[cfg(feature = "fault-injection")]
fn a_poisoned_context_leaks_a_freed_pair_rather_than_pooling_it() {
    // `release_or_leak` is the one place that decides destroy-or-leak, and the
    // pool sits INSIDE its destroy arm rather than beside it. On a poisoned
    // context whose device is not known to be lost, work may still be running
    // against that memory, so the conservative outcome is to leak the pair --
    // pooling it would hand possibly-live memory to the next caller, which is
    // the same wrong answer the retirement boundary exists to prevent. This
    // pins that the pool did not quietly become a third arm of that decision.
    let Some(ctx) = open_or_unmeasured("poisoned free") else {
        return;
    };
    let buf = ctx.buffer(4096).expect("buf");
    ctx.inject_fault(Fault::WaitFails(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    ctx.inject_fault(Fault::IdleFails(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx.download(&buf).expect_err("the download must fail");
    assert!(matches!(err, AckError::Poisoned(_)), "{err}");
    assert!(ctx.is_poisoned());

    let leaked_before = ctx.leaked_on_poison();
    let returns_before = ctx.pool_stats().returns;
    ctx.destroy_buffer(buf);
    assert_eq!(
        ctx.leaked_on_poison(),
        leaked_before + 2,
        "the pair is leaked on purpose, as it was before the pool existed"
    );
    assert_eq!(
        ctx.pool_stats().returns,
        returns_before,
        "and the pool did not take it: the device may still be writing to it"
    );
    assert_eq!(ctx.pooled_objects(), 0);
    // and nothing can be served from it either: every call refuses by name
    assert!(matches!(ctx.buffer(4096), Err(AckError::Poisoned(_))));
}

#[test]
#[cfg(feature = "fault-injection")]
fn an_allocation_that_fails_drains_the_pool_and_retries() {
    // THE POOL MUST NEVER BE THE REASON AN ALLOCATION FAILED. It holds device
    // memory that nothing owns, so a failure with a non-empty pool is the one
    // case where the right answer is certain. This is what makes the ceiling a
    // tuning knob rather than a safety margin, and it is what lets the default
    // be half the device-local heap instead of a quarter -- a quarter BINDS at
    // the registered geometry and costs 17% of the step.
    let Some(ctx) = open_or_unmeasured("reclaim on failure") else {
        return;
    };

    // the control first: with the pool EMPTY the same injected failure must
    // still fail, or the arm below shows nothing
    ctx.set_buffer_pool_enabled(false);
    ctx.set_buffer_pool_enabled(true);
    assert_eq!(ctx.pooled_objects(), 0, "the pool is empty for the control");
    ctx.inject_fault(Fault::CreateFails(
        CreateStep::Memory,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    match ctx.buffer(4096) {
        Err(AckError::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)) => {}
        Err(other) => panic!("wrong error: {other}"),
        Ok(b) => {
            ctx.destroy_buffer(b);
            panic!("with nothing to reclaim the failure must be returned");
        }
    }
    assert_eq!(
        ctx.pool_stats().reclaims,
        0,
        "and nothing was reclaimed, because there was nothing to reclaim"
    );

    // now the arm: put a pair of a DIFFERENT size in the pool, so the request
    // misses and has to allocate, and make that allocation fail
    let held = ctx.buffer(8192).expect("held");
    ctx.destroy_buffer(held);
    assert_eq!(ctx.pooled_objects(), 2, "the pool is holding a pair");
    let reclaims_before = ctx.pool_stats().reclaims;
    ctx.inject_fault(Fault::CreateFails(
        CreateStep::Memory,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    ));
    let b = ctx
        .buffer(4096)
        .expect("the retry after the drain must succeed");
    assert_eq!(
        ctx.pool_stats().reclaims,
        reclaims_before + 1,
        "the failure drained the pool and retried once"
    );
    assert_eq!(
        ctx.pooled_objects(),
        0,
        "and the drain gave the held memory back to the driver"
    );
    assert_eq!(
        ctx.armed_faults(),
        0,
        "the fault was consumed by the first attempt"
    );
    ctx.destroy_buffer(b);
}

#[test]
fn the_ceiling_bounds_what_the_pool_holds() {
    let Some(ctx) = open_or_unmeasured("pool ceiling") else {
        return;
    };
    ctx.set_buffer_pool_max_bytes(4096);
    let a = ctx.buffer(4096).expect("a");
    let b = ctx.buffer(4096).expect("b");
    let declined_before = ctx.pool_stats().declined;
    ctx.destroy_buffer(a);
    ctx.destroy_buffer(b);
    let st = ctx.pool_stats();
    assert_eq!(st.held_bytes, 4096, "the pool holds exactly its ceiling");
    assert_eq!(
        st.declined,
        declined_before + 1,
        "and the pair that would have crossed it was destroyed, not held"
    );
}

#[test]
fn turning_the_pool_off_restores_the_unpooled_object_deltas() {
    // The arm every re-homed leak and retirement assertion runs on. It is a
    // real control: `set_buffer_pool_enabled(false)` drains what is held, so
    // the numbers are the pre-pool ones immediately, not at the next teardown.
    let Some(ctx) = open_or_unmeasured("pool off") else {
        return;
    };
    ctx.set_buffer_pool_enabled(false);
    assert!(!ctx.buffer_pool_enabled());
    let base = ctx.live_objects();
    let b = ctx.buffer(4096).expect("b");
    assert_eq!(ctx.live_objects(), base + 2, "a buffer is two objects");
    let hits_before = ctx.pool_stats().hits;
    ctx.destroy_buffer(b);
    assert_eq!(
        ctx.live_objects(),
        base,
        "back to the baseline: with the pool off a free destroys, exactly as it did before the \
         pool existed"
    );
    assert_eq!(ctx.pooled_objects(), 0);
    let c = ctx.buffer(4096).expect("c");
    assert_eq!(
        ctx.pool_stats().hits,
        hits_before,
        "and nothing is served from a pool that is off"
    );
    ctx.destroy_buffer(c);
}

// ---------------------------------------------------------------------------
// THE FALSIFIER
// ---------------------------------------------------------------------------

/// One arm of the falsifier. Returns `(pool_hits_delta, bytes_read_back)`.
///
/// The fixture: record -- and do NOT flush -- a dispatch whose OUTPUT is
/// `victim`. Free `victim`. Ask for a buffer of the same size. Read it.
///
/// Under the shipped rule the free RETIRES the pair, so the request cannot be
/// served from the pool and the caller gets a genuinely fresh allocation; the
/// pending dispatch then writes into the retired pair, which nobody holds.
///
/// Under `PoolMode::Immediate` the free pools the pair at once, the request is
/// served from it, and the download's flush runs the pending dispatch INTO THE
/// NEW OWNER'S BUFFER. The caller reads a tensor it never wrote and never
/// asked for, with every counter reporting success.
fn recorded_but_unflushed_arm(ctx: &Context, marker: f32) -> (u64, Vec<u8>) {
    let n = 1024u64;
    let shape = [n];
    let src = ctx.buffer(n * 4).expect("src");
    ctx.upload(&src, &f32_bytes(&vec![marker; n as usize]))
        .expect("upload");
    let victim = ctx.buffer(n * 4).expect("victim");
    let kernel = pointwise_kernel(ctx);
    ctx.bind(&kernel, &[&src, &victim, &victim, &victim])
        .expect("bind");
    let (plan, push) = copy_plan(&shape, &src, &victim);
    let access = access_for(&src, &victim);

    // RECORD, do not flush. `max_recorded_items` is 128 by default, so one
    // item leaves the batch open.
    ctx.dispatch(&kernel, &push, plan.dispatch_groups(), 1, &access)
        .expect("record");
    assert_eq!(ctx.recorded_items(), 1, "the batch must still be open");

    ctx.destroy_buffer(victim);

    let hits_before = ctx.pool_stats().hits;
    let fresh = ctx.buffer(n * 4).expect("fresh");
    let hits_delta = ctx.pool_stats().hits - hits_before;

    // the download flushes first, which is what runs the pending dispatch
    let got = ctx.download(&fresh).expect("download");

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(fresh);
    ctx.destroy_buffer(src);
    (hits_delta, got)
}

#[test]
fn a_pooled_pair_is_not_handed_back_while_a_recorded_dispatch_still_names_it() {
    let Some(ctx) = open_or_unmeasured("pool lifetime falsifier") else {
        return;
    };
    let marker = 4321.5f32;
    let n = 1024usize;
    let dispatch_output = f32_bytes(&vec![marker; n]);

    // ---- the shipped arm ---------------------------------------------------
    let (hits, got) = recorded_but_unflushed_arm(&ctx, marker);
    assert_eq!(
        hits, 0,
        "a buffer freed while a batch is open is RETIRED, not pooled: the next request of the \
         same size must not be served from the pool"
    );
    assert_ne!(
        got, dispatch_output,
        "the caller asked for a fresh buffer and must not receive the bytes a pending dispatch \
         was recorded to write somewhere else"
    );

    // ---- the sabotaged arm -------------------------------------------------
    #[cfg(feature = "pool-sabotage")]
    {
        use alelyon_compute_kit::PoolMode;
        ctx.set_pool_mode(PoolMode::Immediate);
        let (sab_hits, sab_got) = recorded_but_unflushed_arm(&ctx, marker);
        ctx.set_pool_mode(PoolMode::Retired);

        // A sabotage that never applied reads as a passing row. Say so.
        if sab_hits != 1 {
            panic!(
                "UNMEASURED (pool lifetime falsifier): PoolMode::Immediate produced {sab_hits} \
                 pool hits, so the sabotage never applied and the shipped arm above is not \
                 evidence of anything"
            );
        }
        assert_eq!(
            sab_got, dispatch_output,
            "UNMEASURED (pool lifetime falsifier): the sabotage was applied but did not DIVERGE. \
             The falsifier is only evidence if handing the pair back early actually produces the \
             wrong answer; if it does not, the shipped arm's assertion is untested"
        );
        assert_ne!(
            got, sab_got,
            "the two arms must differ in the one input that was changed"
        );
    }
    #[cfg(not(feature = "pool-sabotage"))]
    eprintln!(
        "UNMEASURED (pool lifetime falsifier): built without --features pool-sabotage, so only \
         the shipped arm ran. That arm alone cannot show the rule is load-bearing."
    );
}

#[test]
fn a_buffer_freed_while_a_batch_is_open_reaches_the_pool_only_at_the_flush() {
    // The accounting half of the falsifier above, stated as its own row: the
    // pair is held by the retirement list across the open batch and appears in
    // the pool exactly when `release_retired` runs.
    let Some(ctx) = open_or_unmeasured("retirement into the pool") else {
        return;
    };
    let n = 1024u64;
    let shape = [n];
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    let doomed = ctx.buffer(n * 4).expect("doomed");
    ctx.upload(&src, &f32_bytes(&vec![2.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    let (plan, push) = copy_plan(&shape, &src, &dst);

    ctx.dispatch(
        &kernel,
        &push,
        plan.dispatch_groups(),
        1,
        &access_for(&src, &dst),
    )
    .expect("record");
    let returns_before = ctx.pool_stats().returns;
    let pooled_before = ctx.pooled_objects();
    ctx.destroy_buffer(doomed);
    assert_eq!(
        ctx.pool_stats().returns,
        returns_before,
        "recorded work may still name it: the pair is retired, and the pool has not seen it"
    );
    assert_eq!(ctx.pooled_objects(), pooled_before);

    ctx.sync().expect("flush");
    assert_eq!(
        ctx.pool_stats().returns,
        returns_before + 1,
        "the flush waited on its fence, so now nothing can name it and the pool may hold it"
    );
    assert_eq!(
        ctx.pooled_objects(),
        pooled_before + 2,
        "two objects per pair"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn the_retirement_budget_still_flushes_with_the_pool_on() {
    // The pooled counterpart of `retiring_past_the_budget_flushes_the_batch`
    // in tests/batch_flush.rs, which now runs on the un-pooled arm because the
    // pool changes its object delta from 4 to 2. The MECHANISM that test
    // exists for -- `FlushReason::RetiredBytes` firing when a free crosses
    // `max_retired_bytes` -- must survive the pool, and it does only because
    // the pool takes the pair at `release_retired` and not at the free. A pool
    // that intercepted `free_buffer` would leave `BatchPlan::retire` uncalled,
    // and this reason would become unreachable and its budget dead.
    let Some(ctx) = open_or_unmeasured("pooled retirement budget") else {
        return;
    };
    let n = 1024u64;
    let shape = [n];
    let src = ctx.buffer(n * 4).expect("src");
    let dst = ctx.buffer(n * 4).expect("dst");
    ctx.upload(&src, &f32_bytes(&vec![4.0f32; n as usize]))
        .expect("upload");
    let kernel = pointwise_kernel(&ctx);
    let (plan, push) = copy_plan(&shape, &src, &dst);
    ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
    ctx.set_max_retired_bytes(n * 4 / 2);
    let doomed = ctx.buffer(n * 4).expect("doomed");

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
    let flushes_before = ctx.flushes();
    let returns_before = ctx.pool_stats().returns;

    ctx.destroy_buffer(doomed);

    assert_eq!(
        ctx.flushes(),
        flushes_before + 1,
        "crossing the retirement budget still flushes with the pool on"
    );
    assert_eq!(ctx.recorded_items(), 0, "the budget flushed the batch");
    assert_eq!(
        ctx.pool_stats().returns,
        returns_before + 1,
        "and the flush handed the retired pair to the pool"
    );
    assert_eq!(
        ctx.live_objects(),
        objects - 2,
        "across the flush the batch's own fence and command buffer go (2); the retired pair does          NOT, because the pool is holding it -- that is the whole difference from the un-pooled          arm's delta of 4"
    );
    assert_eq!(
        ctx.live_objects(),
        before_batch,
        "net over the whole batch: nothing was destroyed at all"
    );

    ctx.destroy_kernel(kernel);
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
}

#[test]
fn the_pooled_arm_and_the_unpooled_arm_agree_bit_for_bit() {
    // The pool changes WHERE a buffer's memory comes from, not what is
    // computed in it. Same fixture, same context, one input changed.
    let Some(ctx) = open_or_unmeasured("pooled vs unpooled agreement") else {
        return;
    };
    let n = 512u64;
    let shape = [n];
    let values: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();

    let run = |ctx: &Context| -> Vec<u8> {
        let src = ctx.buffer(n * 4).expect("src");
        let dst = ctx.buffer(n * 4).expect("dst");
        ctx.upload(&src, &f32_bytes(&values)).expect("upload");
        let kernel = pointwise_kernel(ctx);
        ctx.bind(&kernel, &[&src, &dst, &dst, &dst]).expect("bind");
        let (plan, push) = copy_plan(&shape, &src, &dst);
        ctx.dispatch(
            &kernel,
            &push,
            plan.dispatch_groups(),
            1,
            &access_for(&src, &dst),
        )
        .expect("dispatch");
        let out = ctx.download(&dst).expect("download");
        ctx.destroy_kernel(kernel);
        ctx.destroy_buffer(src);
        ctx.destroy_buffer(dst);
        out
    };

    ctx.set_buffer_pool_enabled(false);
    let unpooled = run(&ctx);
    ctx.set_buffer_pool_enabled(true);
    let warm = run(&ctx); // fills the pool
    let pooled = run(&ctx); // and now runs off it
    let hits_after_warm = ctx.pool_stats().hits;
    assert!(
        hits_after_warm > 0,
        "the pooled arm must actually have been served from the pool, or it is the same arm twice"
    );
    assert_eq!(unpooled, f32_bytes(&values), "the copy is a copy");
    assert_eq!(warm, unpooled);
    assert_eq!(pooled, unpooled, "the arms agree bit for bit");
}
