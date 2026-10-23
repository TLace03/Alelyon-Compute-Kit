//! The runtime counters at the C ABI (`ack_runtime_stats`, OPTIONAL,
//! 2026-09-29): the batch recorder's flushes by reason and the buffer and
//! staging pools, as `Context` counts them.
//!
//! Every reading here is a DIFFERENCE between two calls on one device, around
//! an action whose effect is known from the recorder's rules. A recorded cast
//! and then a download is one `Readback` flush and nothing else. A recorded
//! cast and then an upload is one `Upload` flush, or none when the recorded
//! upload is on, since the upload then joins the batch. A download with
//! nothing recorded counts nothing. A freed buffer is taken back by the pool
//! at once when no batch names it, and only at the flush when one does; the
//! next allocation of its size is then a hit. The batch depth fires
//! `BatchFull`, which returns without waiting when the asynchronous flush is
//! on. The slots are named by `ack_flush_reason_name`, never by a copy of the
//! list. Without a device each test prints UNMEASURED, unless
//! ACK_REQUIRE_DEVICE is set, in which case it fails.
use alelyon_compute_kit::{ffi, Buffer};
use std::ffi::{c_char, CStr};

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn open_or_unmeasured(what: &str) -> Option<*mut ffi::AckDevice> {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "{what}: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here ({what}): {}", last_error());
        return None;
    }
    Some(dev)
}

fn alloc(dev: *mut ffi::AckDevice, bytes: usize) -> *mut Buffer {
    let b = ffi::ack_buffer_alloc(dev, bytes as u64);
    assert!(!b.is_null(), "{}", last_error());
    b
}

fn free(dev: *mut ffi::AckDevice, b: *mut Buffer) {
    assert_eq!(
        ffi::ack_buffer_free(dev, b),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
}

fn upload(dev: *mut ffi::AckDevice, buf: *const Buffer, bytes: &[u8]) {
    let rc = unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
}

fn download(dev: *mut ffi::AckDevice, buf: *const Buffer, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), out.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    out
}

/// One f32 -> bf16 cast of `n` elements: a RECORDED dispatch, run at the next flush.
fn cast(dev: *mut ffi::AckDevice, src: *const Buffer, dst: *const Buffer, n: usize) {
    assert_eq!(
        ffi::ack_cast(dev, src, dst, n as u64, 0),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
}

fn stats(dev: *mut ffi::AckDevice) -> ffi::AckRuntimeStats {
    let mut s: ffi::AckRuntimeStats = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<ffi::AckRuntimeStats>();
    let rc = unsafe { ffi::ack_runtime_stats(dev, &mut s, size) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_eq!(s.size as usize, size, "the library's struct is this test's");
    assert_eq!(s.version, ffi::ACK_RUNTIME_STATS_VERSION);
    s
}

/// The name of each slot the library uses, read through the C ABI.
fn names(s: &ffi::AckRuntimeStats) -> Vec<String> {
    (0..s.flush_reason_count)
        .map(|i| {
            let p = ffi::ack_flush_reason_name(i);
            assert!(
                !p.is_null(),
                "slot {i} of {} has no name",
                s.flush_reason_count
            );
            unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned()
        })
        .collect()
}

/// The flushes between two readings, by name, the zero ones left out.
fn flushed(a: &ffi::AckRuntimeStats, b: &ffi::AckRuntimeStats) -> Vec<(String, u64)> {
    names(b)
        .into_iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let d = b.flushes_by_reason[i] - a.flushes_by_reason[i];
            (d != 0).then_some((name, d))
        })
        .collect()
}

fn one(name: &str) -> Vec<(String, u64)> {
    vec![(name.to_owned(), 1)]
}

fn dispatch_flushes(dev: *mut ffi::AckDevice) -> u64 {
    let mut s: ffi::AckDispatchStats = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<ffi::AckDispatchStats>();
    let rc = unsafe { ffi::ack_dispatch_stats(dev, &mut s, size) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    s.flushes
}

/// `ACK_DEFERRED_UPLOAD` read by the rule the context read it with at open.
fn deferred_upload_requested() -> bool {
    alelyon_compute_kit::batch::env_flag("ACK_DEFERRED_UPLOAD", false)
}

#[test]
fn each_flush_is_counted_under_its_own_reason_and_only_a_recorded_batch_counts() {
    let Some(dev) = open_or_unmeasured("ack_runtime_stats flushes") else {
        return;
    };
    let n = 4096;
    let src = alloc(dev, n * 4);
    let dst = alloc(dev, n * 2);
    let image: Vec<u8> = (0..n as u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    upload(dev, src, &image);
    let s0 = stats(dev);
    assert_eq!(
        names(&s0),
        [
            "Readback",
            "Upload",
            "TimedDispatch",
            "Close",
            "ExplicitSync",
            "BatchFull",
            "DescriptorsExhausted",
            "RetiredBytes",
            "Accounting"
        ],
        "the recorder's reasons, in its own order"
    );
    assert!(ffi::ack_flush_reason_name(s0.flush_reason_count).is_null());
    assert_eq!(s0.recorded_items, 0, "an upload leaves nothing recorded");
    assert!(s0.max_recorded_items >= 1 && s0.max_retired_bytes > 0);

    // a recorded cast, then a download: one Readback flush, and the dispatch
    // instrument's total agrees
    let total0 = dispatch_flushes(dev);
    cast(dev, src, dst, n);
    let s1 = stats(dev);
    assert_eq!(s1.recorded_items, 1, "the cast is recorded, not submitted");
    assert!(flushed(&s0, &s1).is_empty(), "{:?}", flushed(&s0, &s1));
    download(dev, dst, n * 2);
    let s2 = stats(dev);
    assert_eq!(flushed(&s1, &s2), one("Readback"));
    assert_eq!(s2.recorded_items, 0);
    assert_eq!(s2.batches_in_flight, 0, "a readback waits for its batch");
    assert_eq!(
        dispatch_flushes(dev) - total0,
        1,
        "the two instruments count the same flush"
    );

    // a download with nothing recorded submits no batch and counts nothing
    download(dev, dst, n * 2);
    assert!(flushed(&s2, &stats(dev)).is_empty());

    // a recorded cast, then an upload: one Upload flush, or none when the
    // recorded upload is on and the upload joins the batch
    let s3 = stats(dev);
    cast(dev, src, dst, n);
    upload(dev, src, &image);
    let s4 = stats(dev);
    if deferred_upload_requested() && s4.async_flush == 1 {
        assert!(flushed(&s3, &s4).is_empty(), "{:?}", flushed(&s3, &s4));
    } else {
        assert_eq!(flushed(&s3, &s4), one("Upload"));
    }
    download(dev, dst, n * 2);

    // the batch depth: casts until the recorder submits on its own, which is
    // BatchFull and nothing else, within the depth
    let s5 = stats(dev);
    let mut recorded = 0;
    loop {
        cast(dev, src, dst, n);
        recorded += 1;
        let s = stats(dev);
        if !flushed(&s5, &s).is_empty() {
            assert_eq!(flushed(&s5, &s), one("BatchFull"), "after {recorded} casts");
            assert_eq!(
                s.batches_in_flight, s.async_flush,
                "an asynchronous BatchFull leaves its batch in flight; a synchronous one does not"
            );
            break;
        }
        assert!(
            recorded <= s5.max_recorded_items,
            "{recorded} casts recorded and no flush at depth {}",
            s5.max_recorded_items
        );
    }
    download(dev, dst, n * 2);
    for b in [src, dst] {
        free(dev, b);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}

#[test]
fn a_freed_buffer_returns_to_the_pool_when_no_batch_names_it_and_serves_the_next_allocation() {
    let Some(dev) = open_or_unmeasured("ack_runtime_stats pool") else {
        return;
    };
    // a size nothing else in this context allocates
    let odd = 12_344usize;
    let s0 = stats(dev);
    let a = alloc(dev, odd);
    let s1 = stats(dev);
    if s1.pool_hits == s0.pool_hits && s1.pool_misses == s0.pool_misses {
        // the pool is off for this context (ACK_BUFFER_POOL=0): nothing to hold
        eprintln!("the buffer pool is off here; its counters stay still, as they must");
        free(dev, a);
        assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
        return;
    }
    assert_eq!(
        (s1.pool_misses - s0.pool_misses, s1.pool_hits - s0.pool_hits),
        (1, 0)
    );

    // nothing recorded and nothing in flight: the pool takes the pair back at once
    free(dev, a);
    let s2 = stats(dev);
    assert_eq!(s2.pool_returns - s1.pool_returns, 1);
    assert!(s2.pool_held_objects >= 2 && s2.pool_held_bytes >= odd as u64);
    let b = alloc(dev, odd);
    let s3 = stats(dev);
    assert_eq!(
        (s3.pool_hits - s2.pool_hits, s3.pool_misses - s2.pool_misses),
        (1, 0)
    );

    // a buffer a recorded dispatch reads is RETIRED at its free, not returned:
    // the pool takes it back at the flush
    let n = odd / 4;
    let dst = alloc(dev, n * 2);
    cast(dev, b, dst, n);
    let s4 = stats(dev);
    free(dev, b);
    let s5 = stats(dev);
    assert_eq!(
        s5.pool_returns, s4.pool_returns,
        "retired, not returned, while the batch is open"
    );
    download(dev, dst, n * 2);
    let s6 = stats(dev);
    assert_eq!(
        s6.pool_returns - s5.pool_returns,
        1,
        "the readback's flush took it back"
    );
    let c = alloc(dev, odd);
    let s7 = stats(dev);
    assert_eq!(
        (s7.pool_hits - s6.pool_hits, s7.pool_misses - s6.pool_misses),
        (1, 0)
    );
    for x in [c, dst] {
        free(dev, x);
    }
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
