//! The staging pool: the host-visible staging pair of a COMPLETED transfer is
//! kept mapped and reused by the next transfer of its kind and size class
//! instead of destroyed.
//!
//! Measured on an NVIDIA L4 (driver 580.173.02) under the torch adapter, one
//! A0 training update allocated and freed 612 staging pairs, one per transfer:
//! `vkFreeMemory` took 526 ms of host time and `vkAllocateMemory` most of 206
//! ms more (docs/audits/2026-09-22-ack-nvidia-l4-step-profile.md, section 3c).
//!
//! What may go wrong, and the test that pins each piece:
//!
//! - A pooled pair is LARGER than most transfers it serves (its class is the
//!   next power of two) and holds a previous tenant's bytes. Only the first
//!   `bytes` may ever be copied or read:
//!   `a_smaller_transfer_through_a_larger_pair_moves_only_its_own_bytes`.
//! - A pair is pooled only after the transfer that named it COMPLETED; a
//!   failed wait destroys it: `a_failed_wait_destroys_the_pair_rather_than_
//!   pooling_it`. Recorded uploads hand theirs back at the flush that waited on
//!   them: `recorded_uploads_return_their_pairs_at_the_flush`.
//! - The pool holds host-visible memory nothing owns, so it must never be the
//!   reason a transfer cannot be staged: `a_staging_allocation_failure_drains_
//!   the_pool_and_retries`.
//! - Pooled pairs are LIVE objects: `live_objects()` counts them and
//!   `pooled_objects()` names them; turning the pool off drains it.
//!
//! EVERY TEST ASSERTS THE POOL'S OWN COUNTERS as well as the bytes: a transfer
//! through a reused pair and one through a fresh pair return the same bytes.
//!
//! The pool is OFF by default in the crate and each test turns it on for its
//! own context, with the buffer pool and the fence pool off so every object
//! delta is a staging pair's or a buffer's own. Needs a Vulkan device. Without
//! one every test here prints UNMEASURED and returns, which is a skip with its
//! reason, never a pass; set `ACK_REQUIRE_DEVICE=1` to turn that into a
//! failure on a machine that is supposed to have the device. A GREEN RUN
//! WITHOUT THAT VARIABLE IS NOT DEVICE EVIDENCE.

use alelyon_compute_kit::batch::{
    env_flag, STAGING_POOL_MAX_PAIR_BYTES, STAGING_POOL_MAX_PER_CLASS,
};
use alelyon_compute_kit::{Buffer, Context};

// the recorded-upload test holds six pairs of one class at once
const _: () = assert!(6 <= STAGING_POOL_MAX_PER_CLASS);

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

fn pooled_staging_context(what: &str) -> Option<Context> {
    let ctx = open_or_unmeasured(what)?;
    ctx.set_buffer_pool_enabled(false);
    ctx.set_fence_pool_enabled(false);
    ctx.set_staging_pool_enabled(true);
    assert!(ctx.staging_pool_enabled());
    Some(ctx)
}

/// `n` bytes of a pattern that differs with `seed` at every position.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn upload_new(ctx: &Context, data: &[u8]) -> Buffer {
    let b = ctx.buffer(data.len() as u64).expect("buffer");
    ctx.upload(&b, data).expect("upload");
    b
}

#[test]
fn the_crate_opens_with_the_staging_pool_off_unless_the_environment_turns_it_on() {
    let Some(ctx) = open_or_unmeasured("staging pool default") else {
        return;
    };
    assert_eq!(
        ctx.staging_pool_enabled(),
        env_flag("ACK_STAGING_POOL", false),
        "the crate's default is a pair per transfer; ACK_STAGING_POOL=1 at open is the \
         switch the torch adapter uses"
    );
    assert_eq!(ctx.pooled_staging(), 0, "a fresh context holds no pair");
}

#[test]
fn a_completed_transfer_keeps_its_pair_and_the_next_one_of_its_class_takes_it() {
    let Some(ctx) = pooled_staging_context("staging pool hit") else {
        return;
    };
    let base = ctx.live_objects();
    let n = 4096usize;
    let b = ctx.buffer(n as u64).expect("b");
    assert_eq!(ctx.live_objects(), base + 2, "a buffer is two objects");
    let s0 = ctx.staging_pool_stats();
    assert_eq!((s0.hits, s0.misses, s0.held_pairs), (0, 0, 0));

    let first = pattern(n, 1);
    ctx.upload(&b, &first).expect("upload");
    assert_eq!(ctx.download(&b).expect("download"), first);
    let s1 = ctx.staging_pool_stats();
    assert_eq!(
        (s1.misses, s1.returns, s1.held_pairs),
        (2, 2, 2),
        "an upload's pair and a readback's pair: two kinds, two misses, both kept"
    );
    assert_eq!(
        ctx.live_objects(),
        base + 6,
        "pooled pairs are LIVE objects, two each: live_objects() counts them"
    );
    assert_eq!(
        ctx.pooled_objects(),
        4,
        "and pooled_objects() names them as the pools' share of that count"
    );

    for round in 0..5u8 {
        let v = pattern(n, 10 + round);
        ctx.upload(&b, &v).expect("upload");
        assert_eq!(ctx.download(&b).expect("download"), v, "round {round}");
    }
    let s2 = ctx.staging_pool_stats();
    assert_eq!(
        s2.hits,
        s1.hits + 10,
        "ten transfers of one size, each served from the pool"
    );
    assert_eq!(s2.misses, s1.misses, "and none created a pair");
    assert_eq!(s2.held_pairs, 2);
    assert_eq!(
        ctx.live_objects(),
        base + 6,
        "a hit creates no Vulkan object"
    );

    ctx.destroy_buffer(b);
    assert_eq!(
        ctx.live_objects() - ctx.pooled_objects(),
        base,
        "back to the baseline, once what the pool holds is named"
    );
    assert_eq!(ctx.leaked_on_poison(), 0, "HELD is not LEAKED");
}

#[test]
fn a_smaller_transfer_through_a_larger_pair_moves_only_its_own_bytes() {
    let Some(ctx) = pooled_staging_context("staging pool tenancy") else {
        return;
    };
    // 64 KiB and 40 KiB share the 64 KiB class: the second transfer of each
    // kind is served by the pair the first one filled
    let big = pattern(65536, 0xA5);
    let small = pattern(40960, 0x3C);
    let x = upload_new(&ctx, &big);
    assert_eq!(ctx.download(&x).expect("download"), big);
    let hits = ctx.staging_pool_stats().hits;
    let y = upload_new(&ctx, &small);
    let back = ctx.download(&y).expect("download");
    assert_eq!(
        ctx.staging_pool_stats().hits,
        hits + 2,
        "the small upload and the small readback were both served by the 64 KiB pairs \
         (without the hits this test would not be about reuse)"
    );
    assert_eq!(back.len(), small.len(), "a readback returns its own length");
    assert_eq!(
        back, small,
        "and exactly its own bytes: nothing of the previous tenant's 64 KiB"
    );
    // and the big tenant is untouched by the small one's traffic
    assert_eq!(ctx.download(&x).expect("download"), big);

    // sizes around the class edges: 4 B, 100 B and 4 KiB share the page class;
    // 4097 B takes 8 KiB; 65537 B takes 128 KiB. Each is round-tripped twice,
    // the second time through a pair its first round trip returned.
    for (i, n) in [4usize, 100, 4096, 4097, 65537].into_iter().enumerate() {
        for round in 0..2u8 {
            let v = pattern(n, 50 + 2 * i as u8 + round);
            let b = upload_new(&ctx, &v);
            assert_eq!(
                ctx.download(&b).expect("download"),
                v,
                "{n} B, round {round}"
            );
            ctx.destroy_buffer(b);
        }
    }
    ctx.destroy_buffer(x);
    ctx.destroy_buffer(y);
}

#[test]
fn turning_the_staging_pool_off_drains_it_and_restores_a_pair_per_transfer() {
    let Some(ctx) = pooled_staging_context("staging pool off") else {
        return;
    };
    let base = ctx.live_objects();
    let b = upload_new(&ctx, &pattern(8192, 7));
    assert_eq!(ctx.pooled_staging(), 1);
    assert_eq!(ctx.live_objects(), base + 4);

    ctx.set_staging_pool_enabled(false);
    assert!(!ctx.staging_pool_enabled());
    assert_eq!(
        ctx.pooled_staging(),
        0,
        "turning the pool off destroys what it holds"
    );
    assert_eq!(ctx.staging_pool_stats().held_bytes, 0);
    assert_eq!(ctx.live_objects(), base + 2, "only the buffer is left");
    let v = pattern(8192, 8);
    ctx.upload(&b, &v).expect("upload");
    assert_eq!(ctx.download(&b).expect("download"), v);
    assert_eq!(
        ctx.live_objects(),
        base + 2,
        "with the pool off every transfer creates its pair and destroys it, exactly as \
         before the pool existed"
    );
    ctx.destroy_buffer(b);
    assert_eq!(ctx.live_objects(), base);
}

#[test]
fn recorded_uploads_return_their_pairs_at_the_flush() {
    let Some(ctx) = pooled_staging_context("staging pool, recorded uploads") else {
        return;
    };
    ctx.set_deferred_upload(true);
    if !ctx.deferred_upload() {
        eprintln!(
            "UNMEASURED here: the recorded upload needs the asynchronous flush on and the              timestamps instrument off"
        );
        return;
    }
    let n = 16384usize;
    let buffers: Vec<Buffer> = (0..6).map(|_| ctx.buffer(n as u64).expect("b")).collect();
    for round in 0..3u8 {
        let values: Vec<Vec<u8>> = (0..buffers.len())
            .map(|i| pattern(n, 16 * round + i as u8))
            .collect();
        let hits_before = ctx.staging_pool_stats().hits;
        let readback_pairs = if round == 0 { 0 } else { 1 };
        // six uploads recorded into one open batch: their pairs retire into
        // it, and none may be handed back before the flush that waits on it
        for (b, v) in buffers.iter().zip(&values) {
            ctx.upload(b, v).expect("recorded upload");
        }
        assert_eq!(
            ctx.pooled_staging(),
            readback_pairs,
            "round {round}: the six upload pairs are in the open batch, not in the pool"
        );
        if round > 0 {
            assert_eq!(
                ctx.staging_pool_stats().hits,
                hits_before + 6,
                "round {round}: the six uploads took the six pairs the last round returned"
            );
        }
        for (i, (b, v)) in buffers.iter().zip(&values).enumerate() {
            assert_eq!(
                &ctx.download(b).expect("download"),
                v,
                "round {round}, buffer {i}"
            );
        }
        assert_eq!(
            ctx.pooled_staging(),
            7,
            "round {round}: the readback's flush waited on the batch and its six pairs              came back, beside the one readback pair"
        );
    }
    assert_eq!(ctx.batches_in_flight(), 0);
    for b in buffers {
        ctx.destroy_buffer(b);
    }
}

#[test]
fn the_ceiling_declines_a_pair_that_would_cross_it() {
    let Some(ctx) = pooled_staging_context("staging pool ceiling") else {
        return;
    };
    ctx.set_staging_pool_max_bytes(16384);
    let small = upload_new(&ctx, &pattern(4096, 1));
    assert_eq!(
        ctx.pooled_staging(),
        1,
        "a 4 KiB pair fits under a 16 KiB ceiling"
    );
    let declined = ctx.staging_pool_stats().declined;
    let big = upload_new(&ctx, &pattern(65536, 2));
    let stats = ctx.staging_pool_stats();
    assert_eq!(stats.declined, declined + 1, "a 64 KiB pair would cross it");
    assert_eq!(stats.held_pairs, 1, "and was destroyed, not held");
    assert!(stats.held_bytes <= 16384, "{stats:?}");
    ctx.destroy_buffer(small);
    ctx.destroy_buffer(big);
}

#[test]
fn a_pool_full_of_larger_pairs_evicts_one_for_the_working_set() {
    // THE FAILURE THIS PINS, measured on an NVIDIA L4 through the torch
    // adapter: a model's parameter upload filled the byte ceiling with
    // megabyte-class pairs no training transfer took again, and every later
    // small return was declined -- a pool that was on and pooled nothing.
    let Some(ctx) = pooled_staging_context("staging pool eviction") else {
        return;
    };
    ctx.set_staging_pool_max_bytes(65536);
    // fill it: a 32 KiB upload pair and a 32 KiB readback pair, 64 KiB held
    let big = upload_new(&ctx, &pattern(32768, 1));
    let _ = ctx.download(&big).expect("download");
    let full = ctx.staging_pool_stats();
    assert_eq!((full.held_pairs, full.held_bytes), (2, 65536), "{full:?}");

    // a small transfer's pairs must still be pooled, by evicting the larger
    let small = upload_new(&ctx, &pattern(4096, 2));
    let after = ctx.staging_pool_stats();
    assert_eq!(
        after.evicted,
        full.evicted + 1,
        "one 32 KiB pair made room: {after:?}"
    );
    assert_eq!(
        after.declined, full.declined,
        "and the 4 KiB pair was kept, not declined"
    );
    let v = pattern(4096, 3);
    ctx.upload(&small, &v).expect("upload");
    assert_eq!(ctx.download(&small).expect("download"), v);
    let steady = ctx.staging_pool_stats();
    assert!(
        steady.hits > after.hits,
        "the working set is served from the pool again: {steady:?}"
    );
    assert!(steady.held_bytes <= 65536, "{steady:?}");

    // and a return LARGER than everything held is declined, never evicting
    // the small working set for itself
    let declined = steady.declined;
    let huge = upload_new(&ctx, &pattern(65536, 4));
    let last = ctx.staging_pool_stats();
    assert_eq!(last.declined, declined + 1, "{last:?}");
    assert_eq!(
        last.evicted, steady.evicted,
        "nothing smaller was evicted for it"
    );
    for b in [big, small, huge] {
        ctx.destroy_buffer(b);
    }
}

#[test]
fn a_pair_above_the_size_cap_is_never_pooled() {
    let Some(ctx) = pooled_staging_context("staging pool pair cap") else {
        return;
    };
    let n = (STAGING_POOL_MAX_PAIR_BYTES + 4) as usize;
    let declined = ctx.staging_pool_stats().declined;
    let b = upload_new(&ctx, &pattern(n, 5));
    let stats = ctx.staging_pool_stats();
    assert_eq!(
        stats.declined,
        declined + 1,
        "a pair of {} bytes is above the cap and is destroyed after its transfer",
        n.next_power_of_two()
    );
    assert_eq!(stats.held_pairs, 0, "{stats:?}");
    ctx.destroy_buffer(b);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_failed_wait_destroys_the_pair_rather_than_pooling_it() {
    use alelyon_compute_kit::{AckError, Fault};
    let Some(ctx) = pooled_staging_context("staging pool: failed wait") else {
        return;
    };
    let b = upload_new(&ctx, &pattern(4096, 1));
    assert_eq!(ctx.pooled_staging(), 1, "the first upload's pair is held");
    let live = ctx.live_objects();
    let hits = ctx.staging_pool_stats().hits;

    // the device quiesces after the injected failure, so the submission is
    // released rather than leaked -- but only a wait that SUCCEEDED may pool
    ctx.inject_fault(Fault::WaitFails(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx
        .upload(&b, &pattern(4096, 2))
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
        ctx.staging_pool_stats().hits,
        hits + 1,
        "the failed upload was served by the held pair"
    );
    assert_eq!(
        ctx.pooled_staging(),
        0,
        "and DESTROYED it: a wait that did not succeed does not establish that the pair \
         is free to reuse"
    );
    assert_eq!(
        ctx.live_objects(),
        live - 2,
        "one pair fewer, nothing leaked"
    );
    assert_eq!(ctx.leaked_on_poison(), 0);

    let v = pattern(4096, 3);
    ctx.upload(&b, &v).expect("the context is still usable");
    assert_eq!(ctx.download(&b).expect("download"), v);
    ctx.destroy_buffer(b);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_staging_allocation_failure_drains_the_pool_and_retries() {
    use alelyon_compute_kit::{CreateStep, Fault};
    let Some(ctx) = pooled_staging_context("staging pool: allocation retry") else {
        return;
    };
    // a held pair of ANOTHER class, so the next upload misses and must allocate
    let held = upload_new(&ctx, &pattern(65536, 1));
    assert_eq!(ctx.pooled_staging(), 1);
    let b = ctx.buffer(4096).expect("b");
    let misses = ctx.staging_pool_stats().misses;

    // the upload's own allocation fails once; the pool holds memory nothing
    // owns, so the right answer is certain: give it back and try once more
    ctx.inject_fault(Fault::CreateFails(
        CreateStep::Memory,
        ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    ));
    let v = pattern(4096, 9);
    ctx.upload(&b, &v)
        .expect("the pool was drained and the allocation retried");
    assert_eq!(ctx.armed_faults(), 0, "the injected failure fired");
    assert_eq!(ctx.staging_pool_stats().misses, misses + 1);
    assert_eq!(
        ctx.pooled_staging(),
        1,
        "the 64 KiB pair was drained to make room, and the retried 4 KiB pair was kept"
    );
    assert_eq!(ctx.download(&b).expect("download"), v);
    ctx.destroy_buffer(b);
    ctx.destroy_buffer(held);
}
