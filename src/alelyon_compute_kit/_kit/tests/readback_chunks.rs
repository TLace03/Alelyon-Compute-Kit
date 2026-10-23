//! Chunked readback: `Context::download_into` stages at most
//! `READBACK_CHUNK_BYTES` of host memory per call, copying a larger buffer
//! chunk after chunk through ONE staging pair and straight into the caller's
//! bytes. `download` and the C ABI's `ack_download` read through it.
//!
//! Measured on a Kaggle T4 (driver 580.159.04, 2026-09-27) before the change:
//! `ack_download` read 0.46-0.80 GB/s at 1-64 MiB with the pool off, and the
//! torch adapter (pool on) fell from 2.5 GB/s at 4 MiB to 0.44 GB/s above the
//! pool's largest pair, because a readback staged its whole size in a pair the
//! pool never keeps. CUDA on the same GPU read 10-13 GB/s into pinned memory
//! and 2.8-4.2 GB/s into pageable memory.
//!
//! What may go wrong, and the test that pins each piece:
//!
//! - A chunk lands at the wrong offset, or the tail chunk is dropped or read
//!   past its end. Every size is read back against a pattern that differs at
//!   EVERY 32-bit word of the buffer: a byte pattern of short period would
//!   pass a chunk copied a whole number of periods from its place, and a chunk
//!   is a whole number of any short period. The sizes are one chunk exactly,
//!   a chunk and a word, a chunk and one byte, two and a half chunks, and
//!   three exactly, each with the staging pool off and on:
//!   `every_byte_of_a_readback_lands_in_its_own_place`.
//! - The tail chunk is served by a pair that just held a full chunk. Only its
//!   own `size` bytes may be read: the same test, whose tail chunks are the
//!   smallest of their call.
//! - A readback of any size takes ONE pair of at most one chunk, so with the
//!   pool on a large readback is served from the pool from its second call
//!   on, where its whole-size pair used to be declined:
//!   `a_pooled_context_serves_a_large_readback_through_one_chunk_pair`.
//! - With the pool off the call's one pair is destroyed:
//!   `with_the_pool_off_a_readback_leaves_no_object_behind`.
//! - A destination of the wrong size is refused by name before anything runs,
//!   and the destination is untouched:
//!   `a_destination_of_the_wrong_size_is_refused_and_left_untouched`.
//! - A failed wait destroys the pair rather than pooling it:
//!   `a_failed_wait_destroys_the_chunk_pair_rather_than_pooling_it`
//!   (fault-injection).
//!
//! Each test opens its own context with the buffer pool and the fence pool
//! off, and uploads eagerly (`set_deferred_upload(false)`), so every object
//! and pool delta is the readback's own. Needs a Vulkan device. Without one
//! every test here prints UNMEASURED and returns, which is a skip with its
//! reason, never a pass; set `ACK_REQUIRE_DEVICE=1` to turn that into a
//! failure on a machine that is supposed to have the device. A GREEN RUN
//! WITHOUT THAT VARIABLE IS NOT DEVICE EVIDENCE.

use alelyon_compute_kit::batch::STAGING_POOL_MAX_PAIR_BYTES;
use alelyon_compute_kit::{AckError, Buffer, Context, READBACK_CHUNK_BYTES};

// a chunk's pair must be one the pool keeps: in its own power-of-two class and
// no larger than the pool's largest pair
const _: () = assert!(READBACK_CHUNK_BYTES.is_power_of_two());
const _: () = assert!(READBACK_CHUNK_BYTES <= STAGING_POOL_MAX_PAIR_BYTES);

const CHUNK: usize = READBACK_CHUNK_BYTES as usize;

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

/// A context whose only pool is the staging pool, on or off as asked.
fn context(what: &str, staging_pool: bool) -> Option<Context> {
    let ctx = open_or_unmeasured(what)?;
    ctx.set_buffer_pool_enabled(false);
    ctx.set_fence_pool_enabled(false);
    ctx.set_deferred_upload(false);
    ctx.set_staging_pool_enabled(staging_pool);
    assert_eq!(ctx.staging_pool_enabled(), staging_pool);
    Some(ctx)
}

/// `n` bytes whose every 32-bit word differs from every other word of the
/// buffer: word k is k * 2654435761 + seed, wrapping. The multiplier is odd, so
/// the map is one-to-one over all 2^32 word indices.
fn words(n: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n.next_multiple_of(4));
    for k in 0..n.div_ceil(4) {
        let w = (k as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
        v.extend_from_slice(&w.to_le_bytes());
    }
    v.truncate(n);
    v
}

fn upload_new(ctx: &Context, data: &[u8]) -> Buffer {
    let b = ctx.buffer(data.len() as u64).expect("buffer");
    ctx.upload(&b, data).expect("upload");
    b
}

/// The first byte where `got` and `want` differ, for a message that names the
/// chunk rather than printing megabytes.
fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

#[test]
fn every_byte_of_a_readback_lands_in_its_own_place() {
    let sizes = [
        CHUNK,
        CHUNK + 4,
        CHUNK + 1,
        CHUNK * 5 / 2,
        CHUNK * 3,
        // below one chunk, where nothing is chunked
        4097,
    ];
    for staging_pool in [false, true] {
        let Some(ctx) = context("readback chunks: placement", staging_pool) else {
            return;
        };
        for (i, &n) in sizes.iter().enumerate() {
            let want = words(n, 0x5eed_0000 + i as u32);
            let b = upload_new(&ctx, &want);

            let got = ctx.download(&b).expect("download");
            assert_eq!(got.len(), n);
            assert!(
                got == want,
                "download of {n} bytes (pool {staging_pool}): first difference at byte {:?}, \
                 chunk {:?}",
                first_difference(&got, &want),
                first_difference(&got, &want).map(|at| at / CHUNK)
            );

            // into a destination poisoned first, so a byte the copy never
            // wrote cannot pass as one it wrote
            let mut dst = vec![0xa5u8; n];
            ctx.download_into(&b, &mut dst).expect("download_into");
            assert!(
                dst == want,
                "download_into of {n} bytes (pool {staging_pool}): first difference at byte \
                 {:?}, chunk {:?}",
                first_difference(&dst, &want),
                first_difference(&dst, &want).map(|at| at / CHUNK)
            );
            ctx.destroy_buffer(b);
        }
    }
}

#[test]
fn a_pooled_context_serves_a_large_readback_through_one_chunk_pair() {
    let Some(ctx) = context("readback chunks: pooled", true) else {
        return;
    };
    let n = CHUNK * 3 + 8;
    let want = words(n, 7);
    let b = upload_new(&ctx, &want);
    // the upload's own pair is above the pool's largest and was declined; the
    // readback deltas are measured from here
    let s0 = ctx.staging_pool_stats();

    assert_eq!(ctx.download(&b).expect("first download"), want);
    let s1 = ctx.staging_pool_stats();
    assert_eq!(
        (
            s1.misses - s0.misses,
            s1.returns - s0.returns,
            s1.declined - s0.declined
        ),
        (1, 1, 0),
        "one pair for the whole call, created once and kept: {s0:?} -> {s1:?}"
    );
    assert_eq!(
        s1.held_bytes - s0.held_bytes,
        READBACK_CHUNK_BYTES,
        "the pair is one chunk, not the buffer's {n} bytes"
    );

    let mut dst = vec![0xa5u8; n];
    ctx.download_into(&b, &mut dst).expect("second download");
    assert!(dst == want);
    let s2 = ctx.staging_pool_stats();
    assert_eq!(
        (
            s2.hits - s1.hits,
            s2.misses - s1.misses,
            s2.returns - s1.returns,
            s2.declined - s1.declined
        ),
        (1, 0, 1, 0),
        "the second call is served by the kept pair and hands it back: {s1:?} -> {s2:?}"
    );
    assert_eq!(s2.held_bytes, s1.held_bytes);
    ctx.destroy_buffer(b);
}

#[test]
fn with_the_pool_off_a_readback_leaves_no_object_behind() {
    let Some(ctx) = context("readback chunks: pool off", false) else {
        return;
    };
    let n = CHUNK * 5 / 2;
    let want = words(n, 11);
    let b = upload_new(&ctx, &want);
    let base = ctx.live_objects();
    let s0 = ctx.staging_pool_stats();

    let mut dst = vec![0xa5u8; n];
    ctx.download_into(&b, &mut dst).expect("download_into");
    assert!(dst == want);
    assert_eq!(
        ctx.live_objects(),
        base,
        "the call's one staging pair was destroyed, as every pair is with the pool off"
    );
    assert_eq!(
        ctx.staging_pool_stats(),
        s0,
        "the pool counts nothing while it is off"
    );
    assert_eq!(ctx.pooled_staging(), 0);
    ctx.destroy_buffer(b);
}

#[test]
fn a_destination_of_the_wrong_size_is_refused_and_left_untouched() {
    let Some(ctx) = context("readback chunks: wrong size", true) else {
        return;
    };
    let n = CHUNK + 4;
    let b = upload_new(&ctx, &words(n, 3));
    let base = ctx.live_objects();
    let s0 = ctx.staging_pool_stats();
    for len in [n - 1, n + 1, 1] {
        let mut dst = vec![0xa5u8; len];
        let err = ctx
            .download_into(&b, &mut dst)
            .expect_err("a destination that is not the buffer's size is refused");
        assert!(
            matches!(&err, AckError::Unsupported(m) if m == &format!(
                "download of a {n}-byte buffer into {len} bytes"
            )),
            "{err}"
        );
        assert!(
            dst.iter().all(|&x| x == 0xa5),
            "a refused readback writes nothing into the destination"
        );
    }
    assert_eq!(ctx.live_objects(), base, "nothing was created");
    assert_eq!(ctx.staging_pool_stats(), s0, "nothing was staged");
    assert!(!ctx.is_poisoned());
    ctx.destroy_buffer(b);
}

#[cfg(feature = "fault-injection")]
#[test]
fn a_failed_wait_destroys_the_chunk_pair_rather_than_pooling_it() {
    use alelyon_compute_kit::Fault;
    let Some(ctx) = context("readback chunks: failed wait", true) else {
        return;
    };
    let n = CHUNK * 2 + 4;
    let want = words(n, 5);
    let b = upload_new(&ctx, &want);
    assert_eq!(ctx.download(&b).expect("download"), want);
    let held = ctx.pooled_staging();
    let live = ctx.live_objects();
    let hits = ctx.staging_pool_stats().hits;

    // the device quiesces after the injected failure, so the submission is
    // released rather than leaked -- but only a wait that SUCCEEDED may pool
    ctx.inject_fault(Fault::WaitFails(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let mut dst = vec![0xa5u8; n];
    let err = ctx
        .download_into(&b, &mut dst)
        .expect_err("the injected wait failure surfaces");
    assert!(
        matches!(
            err,
            AckError::Vulkan(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ),
        "{err}"
    );
    assert_eq!(ctx.armed_faults(), 0, "the injected failure fired");
    assert!(!ctx.is_poisoned(), "the device quiesced");
    assert_eq!(
        ctx.staging_pool_stats().hits,
        hits + 1,
        "the failed readback was served by the kept chunk pair"
    );
    assert_eq!(
        ctx.pooled_staging(),
        held - 1,
        "and DESTROYED it: a wait that did not succeed does not establish that the pair is \
         free to reuse"
    );
    assert_eq!(
        ctx.live_objects(),
        live - 2,
        "one pair fewer, nothing leaked"
    );
    assert_eq!(ctx.leaked_on_poison(), 0);

    let mut again = vec![0xa5u8; n];
    ctx.download_into(&b, &mut again)
        .expect("the context is still usable");
    assert!(again == want);
    ctx.destroy_buffer(b);
}
