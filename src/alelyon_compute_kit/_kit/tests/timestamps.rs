//! ACK-L0-04, host-only: two device timestamps resolve to a duration through
//! `resolve_timestamps`, which reduces the difference modulo the counter's
//! width (one wrap) and refuses, by the host clock that bounds the device
//! duration from above, a batch that could have wrapped more than once.
use alelyon_compute_kit::{resolve_timestamps, AckError};

#[test]
fn a_single_wrap_is_resolved_and_a_full_width_counter_never_refuses() {
    // 36 valid bits, 1 ns per tick: the counter wraps every 2^36 ns (~68.7 s)
    let bits = 36;
    let period = 1.0;
    let ns = resolve_timestamps(100, 500, bits, period, 400.0).expect("plain delta");
    assert_eq!(ns, 400.0);
    let near_end = (1u64 << bits) - 10;
    let ns = resolve_timestamps(near_end, 30, bits, period, 40.0)
        .expect("one wrap resolved by the mask");
    assert_eq!(ns, 40.0);
    // a 64-bit counter: no reduction, no refusal, whatever the host time
    let ns = resolve_timestamps(u64::MAX - 5, 4, 64, 0.5, 1.0e12).expect("full width");
    assert_eq!(ns, 5.0);
}

#[test]
fn a_batch_that_could_have_wrapped_more_than_once_is_refused_by_the_host_clock() {
    let bits = 36;
    let period = 1.0;
    let wrap_ns = (1u64 << bits) as f64;
    // the same tick difference as a short batch, but the host waited two periods:
    // the delta is ambiguous and the caller is told so instead of getting 40 ns
    let err = resolve_timestamps((1u64 << bits) - 10, 30, bits, period, 2.0 * wrap_ns + 40.0)
        .expect_err("two wraps cannot be resolved");
    assert!(matches!(err, AckError::AmbiguousTiming(_)), "{err}");
    assert!(err.to_string().contains("ambiguous timing"), "{err}");
    // exactly one period is refused too: the bound is inclusive
    assert!(resolve_timestamps(0, 0, bits, period, wrap_ns).is_err());
    // just under the period is accepted
    assert!(resolve_timestamps(0, 7, bits, period, wrap_ns - 1.0).is_ok());
}
