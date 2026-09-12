//! Deferred submission, layer 0: the decision layer of one recorded batch.
//!
//! Every rule the recorder follows lives here and touches no Vulkan handle, so
//! the rules are testable on a machine with no device (the same shape as
//! `reduce_ops`, `pointwise_ops` and the other plan modules). `Context` owns
//! the handles and the queue; this module owns *when* a batch is submitted,
//! *whether* a barrier precedes an item, *which* descriptor set an item takes,
//! and *whether* a declared access set agrees with what the kernel is bound to.
//!
//! WHAT PHASE 1 RECORDS. Only `Context::dispatch` records. `Context::upload`
//! and `Context::download` flush the recorded batch and then submit their own
//! transfer exactly as they did before, so their error paths, their named
//! refusals and their staging lifetimes are untouched. Recording transfers is
//! follow-on work (see the crate README); nothing here promises it.
//!
//! WHAT IS UNMEASURED. Every constant in this file is an UNMEASURED tuning:
//! no batch depth, no descriptor-pool size and no retirement budget has been
//! measured on any card, because the card belonged to another increment when
//! this was written. Whether deferring submission removes the per-dispatch
//! submit-and-fence cost, and by how much, is UNMEASURED. This module states
//! what the code does and what the tests pin; it makes no claim about any
//! driver's submission or overlap behaviour.
use std::collections::BTreeSet;

/// EVERY reason a recorded batch is submitted.
///
/// This enum is the list: a flush happens for one of these reasons and for no
/// other, because `Context::flush` is the only function in the crate that
/// submits a batch and it takes a `FlushReason`. Adding a submission site
/// means adding a variant, and a new variant that is not enumerated in
/// `ALL_FLUSH_REASONS` fails to compile (`flush_reason_name`'s exhaustive
/// match plus the length assertion below).
///
/// The list is also observable: `Context::flushes_by_reason` counts each one,
/// incremented inside `flush` and nowhere else, so a flush point that stops
/// firing is a failing test rather than a stale sentence in a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FlushReason {
    /// `Context::download`: the copy must see every write recorded before it,
    /// and the host must see the copy. Nothing may be reordered across a
    /// readback, which is what makes deferral safe rather than an optimisation.
    Readback,
    /// `Context::upload`: the copy is submitted eagerly, so it must not run
    /// ahead of dispatches that were recorded before it.
    Upload,
    /// `Context::dispatch_timed`: the duration it returns is a claim about
    /// THAT dispatch. Timing a batch it did not intend to time would be a
    /// measurement of something else.
    TimedDispatch,
    /// `ffi::ack_close`, after the live-buffer refusal: otherwise recorded,
    /// already-counted work is discarded silently and its failure never
    /// surfaces.
    Close,
    /// `Context::sync`: a caller that wants the recorded work to have run
    /// without reading anything back.
    ExplicitSync,
    /// The batch reached `Context::max_recorded_items`, checked AFTER the item
    /// is recorded. With that budget set to 1 the flush fires at the end of
    /// every record call and the deferred path IS the eager arm -- one code
    /// path, no mode branch that can drift.
    BatchFull,
    /// A kernel's descriptor-set watermark reached the pool's capacity. The
    /// sets in use by a recorded, unsubmitted dispatch cannot be rewritten;
    /// only the fence frees them for reuse.
    DescriptorsExhausted,
    /// Device memory retired by `Context::destroy_buffer` while a batch was
    /// open passed `Context::max_retired_bytes`. There is no caching allocator
    /// above this layer -- the torch adapter frees a kit buffer on every
    /// tensor death -- so a step's temporaries retire continuously and without
    /// this budget the batch's held memory grows to the step's working set.
    ///
    /// THE POOL BELOW THIS LAYER DOES NOT MAKE THIS BUDGET DEAD, and that is
    /// a property of where the pool was placed rather than a happy accident.
    /// `Context`'s buffer pool takes a freed pair back at `release_retired`,
    /// which is inside the flush, so `destroy_buffer` still retires and still
    /// calls `BatchPlan::retire`, and this reason still fires. A pool that
    /// intercepted the free instead would leave `retire` uncalled, make this
    /// variant unreachable and turn `max_retired_bytes` into dead code.
    RetiredBytes,
    /// DECLARED AND UNWIRED. The torch adapter has four calls that refuse
    /// while a product is in flight and three that move the accounting
    /// generation (`ack_snapshot`, `ack_counters_reset`, `ack_set_f32_mm_mode`,
    /// `ack_set_bmm_bf16_on_kit`). A frame taken with work recorded and
    /// unsubmitted counts operations that might still fail at the next flush,
    /// and a generation change lets work started in generation g finish in
    /// g+1 -- exactly what the in-flight guard exists to prevent.
    ///
    /// None of those four makes a kit call, so there is NO entry in the C ABI
    /// they could flush through. Wiring this reason needs a new C entry
    /// (`ack_sync`) and therefore an ABI bump, which is not taken here. Until
    /// it is, `is_wired` reports this reason as unwired, `flushes_by_reason`
    /// never moves for it, and the hazard above is real and unguarded.
    Accounting,
}

/// Every variant of `FlushReason`, in declaration order. The index of a reason
/// here is its index in `Context::flushes_by_reason`.
pub const ALL_FLUSH_REASONS: [FlushReason; 9] = [
    FlushReason::Readback,
    FlushReason::Upload,
    FlushReason::TimedDispatch,
    FlushReason::Close,
    FlushReason::ExplicitSync,
    FlushReason::BatchFull,
    FlushReason::DescriptorsExhausted,
    FlushReason::RetiredBytes,
    FlushReason::Accounting,
];

/// A new variant that is not added to `ALL_FLUSH_REASONS` fails to compile
/// here, so the array cannot fall behind the enum.
const _: () = assert!(ALL_FLUSH_REASONS.len() == FLUSH_REASON_COUNT);

/// The number of flush reasons; the width of `Context::flushes_by_reason`.
pub const FLUSH_REASON_COUNT: usize = 9;

/// The stable name of a flush reason. The match is exhaustive on purpose: a
/// new variant breaks this build until it is named and enumerated.
pub fn flush_reason_name(r: FlushReason) -> &'static str {
    match r {
        FlushReason::Readback => "Readback",
        FlushReason::Upload => "Upload",
        FlushReason::TimedDispatch => "TimedDispatch",
        FlushReason::Close => "Close",
        FlushReason::ExplicitSync => "ExplicitSync",
        FlushReason::BatchFull => "BatchFull",
        FlushReason::DescriptorsExhausted => "DescriptorsExhausted",
        FlushReason::RetiredBytes => "RetiredBytes",
        FlushReason::Accounting => "Accounting",
    }
}

/// The index of a reason in `ALL_FLUSH_REASONS` and in the counter array.
pub fn flush_reason_index(r: FlushReason) -> usize {
    match r {
        FlushReason::Readback => 0,
        FlushReason::Upload => 1,
        FlushReason::TimedDispatch => 2,
        FlushReason::Close => 3,
        FlushReason::ExplicitSync => 4,
        FlushReason::BatchFull => 5,
        FlushReason::DescriptorsExhausted => 6,
        FlushReason::RetiredBytes => 7,
        FlushReason::Accounting => 8,
    }
}

/// Whether some code path in this crate can produce this reason.
///
/// `Accounting` is the one that cannot, and the reason is not an oversight: it
/// needs a C entry that does not exist and an ABI bump that was not taken (see
/// the variant's own documentation).
///
/// Of the eight that are wired, the device tests in `tests/batch_flush.rs`
/// fire SEVEN and assert that each moved its own counter and no other:
/// `Readback`, `Upload`, `ExplicitSync`, `TimedDispatch`, `BatchFull`,
/// `DescriptorsExhausted` and `RetiredBytes`. `Close` is not among them and
/// cannot be: `Context::flush` is `pub(crate)` and `Context::sync` hardcodes
/// `ExplicitSync`, so an integration test cannot ask for a `Close` flush, and
/// no C entry exposes this counter, so it cannot be read through the only
/// caller that produces one (`ffi::ack_close`). Its BRANCH is covered instead,
/// through the message a failed close flush leaves in `ack_last_error`; its
/// success path is UNMEASURED at ABI 9.
pub fn is_wired(r: FlushReason) -> bool {
    !matches!(r, FlushReason::Accounting)
}

/// Which items a barrier separates.
///
/// `Declared` (the default since 2026-09-10) records one before an item only
/// when its declared accesses conflict with what the items since the last
/// barrier declared; `Always` (`ACK_BARRIERS=always` at open) records one
/// before every item but the batch's first, which is the rule deferred
/// submission shipped with and the measured control. An item that declares
/// NOTHING takes the barrier under either rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierRule {
    Always,
    Declared,
}

impl BarrierRule {
    pub fn from_env() -> Self {
        match std::env::var("ACK_BARRIERS").ok().as_deref() {
            Some("always") => Self::Always,
            _ => Self::Declared,
        }
    }
}

/// Items recorded into one batch before it is submitted.
///
/// Since the asynchronous flush (2026-09-10) this is the grain at which the
/// host's recording overlaps the device's execution: a `BatchFull` flush
/// submits without waiting and the device runs that batch while the next
/// one is recorded, so the depth has to be SMALLER than the stretch of items
/// between two completing flushes (a readback, an upload) for any overlap to
/// exist. Measured on the registered A2 step (RX 9070 XT, driver 26.8.1,
/// `docs/audits/2026-09-10-compute-kit-host-timeline.md`): 128 gives 7.84 s
/// per update, 256 7.89, 64 8.19 (the per-flush cost), 512 9.97 and 4,096
/// 10.00 (no flush falls inside a stretch, so nothing overlaps), against
/// 10.00 with every flush synchronous. Before that date the flush waited on
/// its fence, this constant was 4,096 (sized so a microbatch was never
/// split), and 256 through 4,096 measured the same. Override with
/// `ACK_MAX_RECORDED_ITEMS`; 1 makes the deferred path the eager arm.
pub const MAX_RECORDED_ITEMS: u32 = 128;

/// Items in the first batch after a completing flush, when the asynchronous
/// flush is on and the timestamps instrument off: the next batch takes twice
/// as many, up to `MAX_RECORDED_ITEMS`. After a readback or an upload the
/// device has nothing queued, and at the depth it would idle until the host
/// had recorded a whole batch; a short first batch starts it sooner and the
/// doubling reaches the depth within three batches (2026-09-10). Override
/// the whole ramp off with `ACK_BATCH_RAMP=0`.
pub const RAMP_FIRST: u32 = 16;

/// Descriptor sets pre-allocated per kernel, all in one `vkAllocateDescriptorSets`
/// at kernel build time.
///
/// Below `MAX_RECORDED_ITEMS` on purpose, so `DescriptorsExhausted` is a
/// reachable, testable branch rather than a dead one. UNMEASURED as a tuning:
/// no family's share of a step's dispatches has been measured against it.
/// Override with `ACK_SETS_PER_KERNEL_POOL`.
pub const SETS_PER_KERNEL_POOL: u32 = 512;

/// Device bytes retired by `destroy_buffer` while a batch is open, before the
/// batch is flushed to destroy them. UNMEASURED as a tuning: no step's
/// temporary working set has been measured against it. Override with
/// `ACK_MAX_RETIRED_BYTES`.
pub const MAX_RETIRED_BYTES: u64 = 1 << 30;

/// The state of one open batch, with no Vulkan handle in it.
///
/// A `BatchPlan` exists only while there is recorded, unsubmitted work: it is
/// created with the first item and consumed by the flush, so `items` is at
/// least 1 for the whole of its life.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    items: u32,
    barriers: u32,
    /// Buffers WRITTEN by the items recorded since the last barrier, and
    /// buffers READ by them. An item conflicts with them when it touches a
    /// buffer they wrote (read-after-write, write-after-write) or writes one
    /// they read (write-after-read); a barrier clears both, because it
    /// separates everything before it from everything after.
    written_since_barrier: BTreeSet<u64>,
    read_since_barrier: BTreeSet<u64>,
    barrier_rule: BarrierRule,
    first_seq: u64,
    next_seq: u64,
    retired_bytes: u64,
    retired_objects: u64,
    max_items: u32,
    max_retired_bytes: u64,
}

impl BatchPlan {
    /// Open a batch whose first item will carry sequence number `first_seq`.
    pub fn new(first_seq: u64, max_items: u32, max_retired_bytes: u64) -> Self {
        Self {
            items: 0,
            barriers: 0,
            written_since_barrier: BTreeSet::new(),
            read_since_barrier: BTreeSet::new(),
            barrier_rule: BarrierRule::from_env(),
            first_seq,
            next_seq: first_seq,
            retired_bytes: 0,
            retired_objects: 0,
            // a budget of 0 would flush before anything could be recorded and
            // never make progress; 1 is the eager arm and the floor
            max_items: max_items.max(1),
            max_retired_bytes,
        }
    }

    /// Account for one item about to be recorded and say whether a barrier
    /// must precede it.
    ///
    /// THE BARRIER RULE. Until 2026-09-10 (R1) exactly one full memory
    /// barrier was recorded immediately before every recorded item except a
    /// batch's first, unconditionally, so no aliasing analysis existed to be
    /// wrong. What that gave up is overlap between INDEPENDENT dispatches,
    /// and the registered pass has many: 3,081 barriers for 3,082 items.
    ///
    /// The rule now consults the DECLARATION each dispatch already carries.
    /// `Context::dispatch` has taken `&[Access]` since deferred submission and
    /// `check_declared_access` refuses one that does not match what the kernel
    /// is bound to, so the declaration is not a hint: a dispatch cannot touch
    /// a buffer it did not declare. A barrier is recorded before an item when
    /// it touches a buffer written since the last barrier (read-after-write,
    /// write-after-write) or writes one read since it (write-after-read), and
    /// a barrier clears both sets because it separates everything before it
    /// from everything after. An item that declares NOTHING takes the barrier.
    /// Buffer ids are unique to memory within a batch -- the pool takes a
    /// freed pair back at the flush, not before -- so two ids never alias.
    /// `ACK_BARRIERS=always` at open restores the unconditional rule; it is
    /// the measured control and both arms produce the same values.
    ///
    /// No barrier separates the last item of a batch from the first item of
    /// the next: under the synchronous flush the fence was a full
    /// serialisation, and under the asynchronous one every batch opens with
    /// one (`Context::open_batch`).
    ///
    /// Since 2026-09-10 the rule is `BarrierRule::Declared`: a barrier is
    /// recorded before an item only when its declared accesses CONFLICT with
    /// what the items since the last barrier declared. `record_with` takes the
    /// declaration; this method is the unconditional rule and is what
    /// `BarrierRule::Always` (`ACK_BARRIERS=always`) restores for every item.
    ///
    /// The invariant of the unconditional rule was arithmetic and testable:
    /// barriers == items - 1. Under the declared rule it is barriers <=
    /// items - 1, with equality when every pair conflicts, which is what a
    /// chain of dependent dispatches is.
    pub fn record(&mut self) -> bool {
        self.record_with(&[])
    }

    /// Record one item that declares `access`, and say whether a barrier must
    /// precede it.
    ///
    /// An EMPTY declaration is treated as a conflict, not as independence: a
    /// caller that says nothing about what it touches gets the unconditional
    /// barrier. That keeps `record()` and every caller that has not been
    /// widened on the old rule, and makes the safe answer the default one.
    pub fn record_with(&mut self, access: &[Access]) -> bool {
        let first = self.items == 0;
        let needs_barrier = !first
            && match self.barrier_rule {
                BarrierRule::Always => true,
                BarrierRule::Declared => access.is_empty() || self.conflicts_with(access),
            };
        if needs_barrier {
            self.barriers += 1;
            self.written_since_barrier.clear();
            self.read_since_barrier.clear();
        }
        for a in access {
            match a.kind {
                AccessKind::Write => self.written_since_barrier.insert(a.buffer),
                AccessKind::Read => self.read_since_barrier.insert(a.buffer),
            };
        }
        self.items += 1;
        self.next_seq += 1;
        needs_barrier
    }

    /// Whether `access` has a hazard against the items since the last barrier:
    /// it touches a buffer they wrote, or writes one they read.
    fn conflicts_with(&self, access: &[Access]) -> bool {
        access.iter().any(|a| {
            self.written_since_barrier.contains(&a.buffer)
                || (a.kind == AccessKind::Write && self.read_since_barrier.contains(&a.buffer))
        })
    }

    /// Record a transfer whose copy WRITES `buffer` (a recorded upload), and
    /// say whether a barrier must precede it. It is not an item of the plan.
    pub fn record_transfer_write(&mut self, buffer: u64) -> bool {
        let needs_barrier = self.items > 0
            && match self.barrier_rule {
                BarrierRule::Always => true,
                BarrierRule::Declared => {
                    self.written_since_barrier.contains(&buffer)
                        || self.read_since_barrier.contains(&buffer)
                }
            };
        if needs_barrier {
            self.written_since_barrier.clear();
            self.read_since_barrier.clear();
        }
        self.written_since_barrier.insert(buffer);
        needs_barrier
    }

    /// The reason this batch must be flushed now, checked AFTER an item was
    /// recorded. `None` means recording continues.
    pub fn due(&self) -> Option<FlushReason> {
        if self.items >= self.max_items {
            return Some(FlushReason::BatchFull);
        }
        None
    }

    /// Account for objects retired into this batch (they are destroyed when
    /// its fence signals) and say whether the retirement budget now demands a
    /// flush.
    ///
    /// A buffer retires two objects holding its device bytes; a kernel retires
    /// four holding none, so a kernel can never itself cross the budget. The
    /// budget is on BYTES, so `objects` is accounting only.
    pub fn retire(&mut self, bytes: u64, objects: u64) -> Option<FlushReason> {
        self.retired_bytes = self.retired_bytes.saturating_add(bytes);
        self.retired_objects = self.retired_objects.saturating_add(objects);
        if self.retired_bytes > self.max_retired_bytes {
            return Some(FlushReason::RetiredBytes);
        }
        None
    }

    pub fn items(&self) -> u32 {
        self.items
    }

    pub fn barriers(&self) -> u32 {
        self.barriers
    }

    pub fn first_seq(&self) -> u64 {
        self.first_seq
    }

    /// One past the sequence number of the last recorded item.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn retired_bytes(&self) -> u64 {
        self.retired_bytes
    }

    /// Vulkan objects retired into this batch (a buffer retires two: the
    /// buffer and its memory; a kernel retires four: its pool, pipeline,
    /// pipeline layout and descriptor set layout).
    pub fn retired_objects(&self) -> u64 {
        self.retired_objects
    }

    /// The batch's own description, for the message a flush failure carries.
    pub fn describe(&self, reason: FlushReason) -> String {
        format!(
            "reason {}, {} recorded dispatch(es), {} barrier(s), sequence {}..{}, {} retired object(s) holding {} byte(s)",
            flush_reason_name(reason),
            self.items,
            self.barriers,
            self.first_seq,
            self.next_seq.saturating_sub(1),
            self.retired_objects,
            self.retired_bytes,
        )
    }
}

/// Whether a declared buffer access reads or writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessKind {
    Read,
    Write,
}

/// One buffer a dispatch declares it touches, named by `Buffer::id`.
///
/// A SLOT THE OPERATION DOES NOT USE IS DECLARED `Read`. Several families bind
/// a live buffer into a slot the operation never reads (the row family's
/// `gamma` slot, the loss family's `upstream` slot); declaring those `Read` is
/// conservative -- it can only keep a barrier that could have been dropped,
/// never drop one that was needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Access {
    pub buffer: u64,
    pub kind: AccessKind,
}

impl Access {
    pub fn read(buffer: u64) -> Self {
        Self {
            buffer,
            kind: AccessKind::Read,
        }
    }

    pub fn write(buffer: u64) -> Self {
        Self {
            buffer,
            kind: AccessKind::Write,
        }
    }
}

/// The declared access set must name the kernel's bound buffers, slot for
/// slot, in order.
///
/// IN PHASE 1 THE ACCESS SETS DECIDE NOTHING. R1 records a barrier between
/// every pair of adjacent items whatever they touch, so no analysis reads
/// these declarations. They are here for two reasons and neither of them is an
/// analysis: they are a second, independent statement of what each C entry
/// binds -- checked here against the binding itself, so a call site that
/// declares one thing and binds another is refused by name rather than
/// silently disagreeing -- and they are the wiring barrier elision (R2) would
/// otherwise have to add to nine call sites with no test coverage in between.
/// Nobody reading this should believe an aliasing analysis is running.
pub fn check_declared_access(declared: &[Access], bound: &[u64]) -> Result<(), String> {
    if declared.len() != bound.len() {
        return Err(format!(
            "dispatch declares {} buffer access(es), the kernel is bound to {} buffer(s)",
            declared.len(),
            bound.len()
        ));
    }
    for (slot, (a, b)) in declared.iter().zip(bound.iter()).enumerate() {
        if a.buffer != *b {
            return Err(format!(
                "dispatch declares buffer {} in slot {slot}, the kernel is bound to buffer {b} there",
                a.buffer
            ));
        }
    }
    Ok(())
}

/// The next descriptor set index a kernel hands out, or `None` when its pool's
/// pre-allocated sets are all held by recorded, unsubmitted work.
///
/// The watermark is reset by a flush and by nothing else: a set written for a
/// recorded dispatch may not be rewritten until that dispatch has run, which
/// is the wrong-answer hazard this arena exists to remove. Before this change
/// a kernel had exactly one descriptor set, and rebinding it between two
/// dispatches was safe only because each dispatch was submitted and waited for
/// on its own.
pub fn next_set_index(watermark: u32, capacity: u32) -> Option<u32> {
    if watermark >= capacity {
        return None;
    }
    Some(watermark)
}

/// Read a `u32` tuning from the environment, falling back to `default` for an
/// absent, unparseable or zero value. Zero is rejected rather than accepted
/// because every one of these budgets must admit at least one item.
pub fn env_u32(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(v) => v
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(default),
        Err(_) => default,
    }
}

/// Read a `u64` tuning from the environment (see `env_u32`).
pub fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|n| *n > 0)
            .unwrap_or(default),
        Err(_) => default,
    }
}

/// Read an on/off tuning from the environment.
///
/// NOT `env_u32`: that one filters out zero and falls back to its default, so
/// it cannot express "off". A switch whose "0" silently meant "on" would make
/// every un-pooled control arm a second copy of the pooled arm.
pub fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim() {
            "0" | "false" | "off" | "no" => false,
            "" => default,
            _ => true,
        },
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reason_array_covers_the_enum_and_the_names_are_distinct() {
        assert_eq!(ALL_FLUSH_REASONS.len(), FLUSH_REASON_COUNT);
        let mut names: Vec<&str> = ALL_FLUSH_REASONS
            .iter()
            .map(|r| flush_reason_name(*r))
            .collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            before,
            "two flush reasons share a name: {names:?}"
        );
        for (i, r) in ALL_FLUSH_REASONS.iter().enumerate() {
            assert_eq!(
                flush_reason_index(*r),
                i,
                "{} is not at its own index",
                flush_reason_name(*r)
            );
        }
    }

    #[test]
    fn accounting_is_the_only_unwired_reason_and_that_is_a_pinned_fact() {
        // The list of flush points is a promise. This test is what keeps the
        // one promise the crate does NOT keep from being forgotten: the
        // accounting flush point is declared and unwired because it needs a C
        // entry that does not exist and an ABI bump that was not taken. If
        // somebody wires it, this test fails and they must say so here.
        let unwired: Vec<&str> = ALL_FLUSH_REASONS
            .iter()
            .filter(|r| !is_wired(**r))
            .map(|r| flush_reason_name(*r))
            .collect();
        assert_eq!(
            unwired,
            vec!["Accounting"],
            "exactly one flush reason is unwired, and it is the accounting one"
        );
    }

    #[test]
    fn a_batch_records_one_barrier_between_every_pair_of_adjacent_items() {
        for n in 0u32..64 {
            let mut plan = BatchPlan::new(0, MAX_RECORDED_ITEMS, MAX_RETIRED_BYTES);
            for _ in 0..n {
                plan.record();
            }
            assert_eq!(plan.items(), n);
            assert_eq!(
                plan.barriers(),
                n.saturating_sub(1),
                "n = {n}: barriers must be items - 1"
            );
        }
    }

    #[test]
    fn the_first_item_of_a_batch_takes_no_barrier_and_every_later_one_does() {
        let mut plan = BatchPlan::new(0, MAX_RECORDED_ITEMS, MAX_RETIRED_BYTES);
        assert!(!plan.record(), "the first item of a batch needs no barrier");
        for i in 1..10 {
            assert!(plan.record(), "item {i} must be preceded by a barrier");
        }
    }

    #[test]
    fn batch_full_fires_at_the_nth_item_and_not_at_the_n_plus_first() {
        for max in [1u32, 2, 3, 17, 4096] {
            let mut plan = BatchPlan::new(0, max, MAX_RETIRED_BYTES);
            for i in 1..max {
                plan.record();
                assert_eq!(plan.due(), None, "max {max}: item {i} is not the limit");
            }
            plan.record();
            assert_eq!(
                plan.due(),
                Some(FlushReason::BatchFull),
                "max {max}: the limit is reached AT the {max}th item, not after it"
            );
        }
    }

    #[test]
    fn a_budget_of_one_item_makes_the_deferred_path_the_eager_arm() {
        // Not a second implementation with a mode branch that can drift: the
        // same code path, with the budget set to its floor, flushes at the end
        // of every record call.
        let mut plan = BatchPlan::new(0, 1, MAX_RETIRED_BYTES);
        plan.record();
        assert_eq!(plan.due(), Some(FlushReason::BatchFull));
        // and a budget of 0 is clamped to the floor rather than wedging
        let mut zero = BatchPlan::new(0, 0, MAX_RETIRED_BYTES);
        zero.record();
        assert_eq!(zero.due(), Some(FlushReason::BatchFull));
    }

    #[test]
    fn retiring_past_the_budget_asks_for_a_flush_and_below_it_does_not() {
        let mut plan = BatchPlan::new(0, MAX_RECORDED_ITEMS, 1000);
        plan.record();
        assert_eq!(plan.retire(400, 2), None);
        assert_eq!(
            plan.retire(600, 2),
            None,
            "exactly at the budget is not past it"
        );
        assert_eq!(plan.retired_bytes(), 1000);
        assert_eq!(plan.retired_objects(), 4);
        assert_eq!(plan.retire(1, 2), Some(FlushReason::RetiredBytes));
        assert_eq!(plan.retired_objects(), 6, "a buffer retires two objects");
    }

    #[test]
    fn the_retirement_accounting_saturates_rather_than_wrapping() {
        let mut plan = BatchPlan::new(0, MAX_RECORDED_ITEMS, u64::MAX);
        plan.record();
        assert_eq!(plan.retire(u64::MAX, 2), None);
        assert_eq!(plan.retire(u64::MAX, 2), None);
        assert_eq!(
            plan.retired_bytes(),
            u64::MAX,
            "saturating, not wrapped to a small number"
        );
    }

    #[test]
    fn the_sequence_range_names_the_items_this_batch_recorded() {
        let mut plan = BatchPlan::new(18_432, MAX_RECORDED_ITEMS, MAX_RETIRED_BYTES);
        for _ in 0..1204 {
            plan.record();
        }
        assert_eq!(plan.first_seq(), 18_432);
        assert_eq!(plan.next_seq(), 18_432 + 1204);
        let text = plan.describe(FlushReason::Readback);
        assert!(text.contains("Readback"), "{text}");
        assert!(text.contains("18432..19635"), "{text}");
        assert!(text.contains("1204 recorded dispatch"), "{text}");
    }

    #[test]
    fn a_declared_access_set_must_name_the_bound_buffers_slot_for_slot() {
        let bound = vec![7u64, 9, 7, 11];
        // the shape the row family produces: a slot the operation does not
        // read is bound to a live buffer and declared Read
        let ok = [
            Access::read(7),
            Access::read(9),
            Access::read(7),
            Access::write(11),
        ];
        assert_eq!(check_declared_access(&ok, &bound), Ok(()));

        let missing = [Access::read(7), Access::read(9), Access::write(11)];
        let err = check_declared_access(&missing, &bound).expect_err("a missing slot is refused");
        assert!(err.contains("declares 3"), "{err}");
        assert!(err.contains("bound to 4"), "{err}");

        let extra = [
            Access::read(7),
            Access::read(9),
            Access::read(7),
            Access::write(11),
            Access::read(11),
        ];
        assert!(
            check_declared_access(&extra, &bound).is_err(),
            "an extra slot is refused"
        );

        let wrong = [
            Access::read(7),
            Access::read(9),
            Access::read(9),
            Access::write(11),
        ];
        let err = check_declared_access(&wrong, &bound).expect_err("a wrong buffer is refused");
        assert!(err.contains("slot 2"), "the refusal names the slot: {err}");

        // a kernel with no storage buffers declares nothing
        assert_eq!(check_declared_access(&[], &[]), Ok(()));
    }

    #[test]
    fn the_kind_of_a_declared_access_changes_nothing_in_phase_one() {
        // Stated as a test so nobody reads the Access type and believes an
        // analysis is running: swapping every Read for a Write is accepted,
        // because R1's barrier is unconditional and nothing reads the kind.
        let bound = vec![1u64, 2];
        let reads = [Access::read(1), Access::read(2)];
        let writes = [Access::write(1), Access::write(2)];
        assert_eq!(check_declared_access(&reads, &bound), Ok(()));
        assert_eq!(check_declared_access(&writes, &bound), Ok(()));
    }

    #[test]
    fn a_descriptor_arena_hands_out_every_set_once_and_then_refuses() {
        for capacity in [1u32, 2, 512] {
            for i in 0..capacity {
                assert_eq!(next_set_index(i, capacity), Some(i), "capacity {capacity}");
            }
            assert_eq!(
                next_set_index(capacity, capacity),
                None,
                "capacity {capacity}: the watermark at capacity is exhaustion"
            );
            // and a reset (what a flush does) makes the first set available again
            assert_eq!(next_set_index(0, capacity), Some(0));
        }
    }

    #[test]
    fn the_environment_tunings_reject_zero_and_garbage_and_keep_the_default() {
        // A zero budget would admit no item; garbage must not silently become
        // a different number. Both fall back to the declared default.
        let name = "ACK_BATCH_TEST_TUNING_THAT_DOES_NOT_EXIST";
        std::env::remove_var(name);
        assert_eq!(env_u32(name, 4096), 4096);
        assert_eq!(env_u64(name, 1 << 30), 1 << 30);
        std::env::set_var(name, "0");
        assert_eq!(env_u32(name, 4096), 4096, "zero is not a budget");
        std::env::set_var(name, "not a number");
        assert_eq!(env_u32(name, 4096), 4096);
        std::env::set_var(name, " 7 ");
        assert_eq!(
            env_u32(name, 4096),
            7,
            "a real value wins, whitespace and all"
        );
        std::env::remove_var(name);
    }
}
