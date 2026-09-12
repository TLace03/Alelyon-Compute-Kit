//! Alelyon Compute Kit (ACK), layer 0: a thin, explicit Vulkan 1.4 compute
//! runtime. It owns one logical device with one compute queue, device-local
//! buffers with staging transfers, compute pipelines built from SPIR-V, and
//! GPU-timestamped dispatch. Nothing here is a tensor library: layer 0 exists
//! so the kernels above it (`kernels/`) can be measured on any Vulkan 1.4
//! device before anything is built on them.
//!
//! What this file DECLARES about the device is read back from the driver
//! (`DeviceReport`); what it OBSERVES is the timestamp delta of a dispatch.

use ash::vk;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::io::Cursor;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod attention_ops;
pub mod batch;
pub mod embedding_ops;
pub mod ffi;
pub mod loss_ops;
pub mod matmul_ops;
pub mod packed2_ops;
pub mod pointwise_ops;
pub mod reduce_ops;
pub mod row_ops;
pub mod vq_ops;

pub use batch::{Access, AccessKind, FlushReason, ALL_FLUSH_REASONS, FLUSH_REASON_COUNT};

/// Milestone-1 result of asking the driver what it has.
#[derive(Debug, Clone)]
pub struct DeviceReport {
    pub device_name: String,
    pub api_version: String,
    pub driver_version_raw: u32,
    pub driver_name: String,
    pub driver_info: String,
    pub subgroup_size: u32,
    pub timestamp_period_ns: f32,
    pub bf16_type: bool,
    pub bf16_dot_product: bool,
    pub bf16_cooperative_matrix: bool,
    pub cooperative_matrix: bool,
    /// `shaderInt16`: the f32 -> bf16 cast kernel, built at every C-ABI open,
    /// declares the SPIR-V `Int16` capability (a review decoded every module:
    /// it is the only one), so `open` requires and enables the feature.
    pub shader_int16: bool,
    pub coopmat_shapes: Vec<CoopmatShape>,
    pub device_local_bytes: u64,
    /// Identity, as the driver declares it (readiness ledger, "ABI/capability
    /// handshake"): the PCI ids, the packed API version, the driver id and the
    /// two UUIDs from VK_KHR_external_memory_capabilities (core 1.1).
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub api_version_raw: u32,
    pub driver_id: i32,
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
    /// Compute limits beyond the ones `Context` checks against.
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
}

impl DeviceReport {
    /// The bytes a stable device identity is hashed from: PCI vendor and
    /// device ids, the device and driver UUIDs, the raw driver version and the
    /// packed API version. Two opens of one device with one driver give the
    /// same bytes; a driver update or another card changes them.
    pub fn identity_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(48);
        out.extend_from_slice(&self.vendor_id.to_le_bytes());
        out.extend_from_slice(&self.device_id.to_le_bytes());
        out.extend_from_slice(&self.device_uuid);
        out.extend_from_slice(&self.driver_uuid);
        out.extend_from_slice(&self.driver_version_raw.to_le_bytes());
        out.extend_from_slice(&self.api_version_raw.to_le_bytes());
        out
    }
}

/// One row of VK_KHR_cooperative_matrix properties, decoded.
#[derive(Debug, Clone)]
pub struct CoopmatShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: String,
    pub b_type: String,
    pub c_type: String,
    pub result_type: String,
    pub scope: String,
    pub saturating: bool,
}

/// VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: read from
/// vulkan_core.h 1.4.341 (ash 0.38 tracks 1.3.281 and predates the extension).
const STRUCTURE_TYPE_SHADER_BFLOAT16_FEATURES_KHR: i32 = 1000141000;
const COMPONENT_TYPE_BFLOAT16_KHR: i32 = 1000141000;

/// Manual mirror of VkPhysicalDeviceShaderBfloat16FeaturesKHR (vulkan_core.h:11185).
#[repr(C)]
struct ShaderBfloat16FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

impl Default for ShaderBfloat16FeaturesKHR {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(STRUCTURE_TYPE_SHADER_BFLOAT16_FEATURES_KHR),
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        }
    }
}

fn component_type_name(t: vk::ComponentTypeKHR) -> String {
    match t.as_raw() {
        0 => "float16".to_string(),
        1 => "float32".to_string(),
        2 => "float64".to_string(),
        3 => "sint8".to_string(),
        4 => "sint16".to_string(),
        5 => "sint32".to_string(),
        6 => "sint64".to_string(),
        7 => "uint8".to_string(),
        8 => "uint16".to_string(),
        9 => "uint32".to_string(),
        10 => "uint64".to_string(),
        COMPONENT_TYPE_BFLOAT16_KHR => "bfloat16".to_string(),
        other => format!("component_type_{other}"),
    }
}

fn scope_name(s: vk::ScopeKHR) -> String {
    match s.as_raw() {
        1 => "device".to_string(),
        2 => "workgroup".to_string(),
        3 => "subgroup".to_string(),
        5 => "queue_family".to_string(),
        other => format!("scope_{other}"),
    }
}

#[derive(Debug)]
pub enum AckError {
    Load(String),
    Vulkan(vk::Result),
    NoDevice(String),
    Unsupported(String),
    Io(String),
    /// A buffer or kernel handed to a context that did not create it.
    Foreign(String),
    /// A buffer used, or still bound to a kernel, after it was destroyed.
    Freed(String),
    /// The context's device state is unknown after a failed wait and a failed
    /// idle, or the device is lost; every call refuses until the context is
    /// dropped (review finding ACK-L0-03).
    Poisoned(String),
    /// A timed dispatch completed but its device duration cannot be resolved:
    /// the host clock says the batch ran at or beyond the timestamp counter's
    /// period, so the tick difference is ambiguous (ACK-L0-04). The work is
    /// done; only the timing is unknown.
    AmbiguousTiming(String),
    /// A flush of the deferred batch failed before its submission succeeded.
    /// The message names the batch -- its flush reason, its item count and its
    /// sequence range -- because the failure belongs to the batch and not to
    /// whichever call happened to trigger the flush. It carries no C code of
    /// its own: `ffi::code_for` maps it through the same arm as any other
    /// Vulkan failure, `ACK_ERR_VULKAN`.
    Batch(String),
}

impl From<vk::Result> for AckError {
    fn from(r: vk::Result) -> Self {
        AckError::Vulkan(r)
    }
}

impl std::fmt::Display for AckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AckError::Load(s) => write!(f, "vulkan loader: {s}"),
            AckError::Vulkan(r) => write!(f, "vulkan call failed: {r:?}"),
            AckError::NoDevice(s) => write!(f, "no usable device: {s}"),
            AckError::Unsupported(s) => write!(f, "unsupported: {s}"),
            AckError::Io(s) => write!(f, "io: {s}"),
            AckError::Foreign(s) => write!(f, "foreign resource: {s}"),
            AckError::Freed(s) => write!(f, "freed resource: {s}"),
            AckError::Poisoned(s) => write!(f, "poisoned context: {s}"),
            AckError::AmbiguousTiming(s) => write!(f, "ambiguous timing: {s}"),
            AckError::Batch(s) => write!(f, "deferred batch: {s}"),
        }
    }
}

impl std::error::Error for AckError {}

pub type Result<T> = std::result::Result<T, AckError>;

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

/// One logical device, one compute queue, one command pool.
///
/// **Ownership contract (review finding ACK-L0-02).** Every `Buffer` and
/// `Kernel` carries the id of the context that created it, and every method
/// that takes one refuses a resource from another context before any Vulkan
/// call (`AckError::Foreign`). `bind` records which buffers a kernel holds and
/// `dispatch_timed` refuses if one of them was destroyed since
/// (`AckError::Freed`), so a descriptor can never point at freed memory.
///
/// **Thread contract.** A `Context` is `Send` and deliberately not `Sync`: one
/// queue, one command pool and one descriptor set per kernel are externally
/// synchronised Vulkan objects, so sharing a context between threads needs a
/// lock around it (the C ABI keeps each device behind a mutex). The compiler
/// enforces this (E0277, "`Context` cannot be shared between threads safely"):
///
/// ```compile_fail,E0277
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<alelyon_compute_kit::Context>();
/// ```
///
/// and the same path resolves for the half that must compile, `Send`:
///
/// ```
/// fn assert_send<T: Send>() {}
/// assert_send::<alelyon_compute_kit::Context>();
/// ```
pub struct Context {
    /// Kept alive on purpose when the context is poisoned: dropping it would
    /// unload the Vulkan loader under an instance that is deliberately leaked.
    _entry: std::mem::ManuallyDrop<ash::Entry>,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    command_pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub report: DeviceReport,
    /// Vulkan 1.3 subgroupSizeControl + computeFullSubgroups were enabled.
    pub subgroup_size_control: bool,
    /// Valid bits of the compute queue's timestamps (36..=64); deltas are
    /// reduced modulo 2^bits (review finding ACK-L0-04).
    pub timestamp_valid_bits: u32,
    /// Device limits the safe API checks against (review finding ACK-L0-01).
    pub max_storage_buffer_bytes: u64,
    pub max_workgroup_count: [u32; 3],
    pub max_push_constant_bytes: u32,
    /// Identity of this context; resources carry it (ACK-L0-02).
    id: u64,
    /// Ids of the buffers this context created and has not destroyed.
    live_buffers: RefCell<HashSet<u64>>,
    next_buffer: Cell<u64>,
    /// `Cell` is `Send` and not `Sync`: the marker that keeps a `Context` from
    /// being shared between threads without a lock.
    _not_sync: PhantomData<Cell<()>>,
    /// Set when a submission's fence wait failed and the device could not be
    /// quiesced (or was lost); read by every method first (ACK-L0-03).
    poisoned: RefCell<Option<String>>,
    /// Device loss is a poison whose teardown is permitted; any other poison
    /// leaks its objects rather than destroy them under possibly running work.
    lost: Cell<bool>,
    /// Vulkan objects deliberately left allocated because the context is
    /// poisoned and the device is not known to be lost: a fence, a command
    /// buffer, a staging buffer and its memory, a query pool, a buffer and its
    /// memory, each of a kernel's four objects (one count per object). When
    /// the failure is a deferred batch's flush, the set is the BATCH's: its
    /// fence, its command buffer and every buffer a caller freed into its
    /// retirement list.
    leaked: Cell<u64>,
    /// Submissions whose fence and command buffer were released (tests read
    /// it). A flush of a recorded batch releases ONE fence and ONE command
    /// buffer for the whole batch, so this counts submissions, which is no
    /// longer the same number as operations.
    torn_down: Cell<u64>,
    /// Vulkan objects created by this context's methods and not yet destroyed
    /// (buffers, memories, shader modules, layouts, pipelines, descriptor
    /// pools, fences, command buffers, query pools), counted by the create and
    /// destroy wrappers themselves (ACK-L0-05). Leaked-on-poison objects stay
    /// counted: they are live. So are objects RETIRED into an open batch: a
    /// buffer destroyed while recorded work may still reference it is counted
    /// until the flush that destroys it.
    live_objects: Cell<i64>,
    /// The open deferred batch, or `None` when nothing is recorded. A
    /// `Recorder` exists only while it holds at least one recorded item, so
    /// `Some` means there is unsubmitted work and `None` means there is not.
    /// Allocated lazily at the first recorded item, so a fresh context still
    /// counts zero live objects.
    recorder: RefCell<Option<Recorder>>,
    /// One counter per `FlushReason`, incremented inside `flush` and nowhere
    /// else (`flushes_by_reason`).
    flushes: [Cell<u64>; batch::FLUSH_REASON_COUNT],
    /// Moved by every flush, including a flush of an empty batch. A kernel
    /// whose descriptor watermark was last touched in an earlier generation
    /// starts again at set 0, which is how the arena is reset without the
    /// context holding a back-pointer to every kernel the caller owns.
    batch_gen: Cell<u64>,
    /// The descriptor bank (0 or 1) the open batch's kernels hand sets out of:
    /// the one the batch in flight is not using (asynchronous flushes,
    /// 2026-09-10). A kernel's arena holds two banks so a batch can be
    /// recorded while the previous one executes without rewriting a set the
    /// device is still reading.
    bank: Cell<usize>,
    /// The batch the last non-completing flush submitted and nothing has
    /// finished yet: its fence is waited on by the next flush, or at once by
    /// a flush whose reason promises completion (`completes`). At most one.
    in_flight: RefCell<Option<Submitted>>,
    /// `ACK_ASYNC_FLUSH=0` at open makes every flush complete before it returns:
    /// the measured control for the asynchronous flush, not a contract.
    async_flush: Cell<bool>,
    /// Batches submitted without completing since the last completing flush.
    /// The batch opened after a completing flush is short (`RAMP_FIRST`
    /// items), each next one twice as long up to the depth, so the device
    /// starts on the stretch after a sync point while the host is still
    /// recording it (2026-09-10). `ACK_BATCH_RAMP=0` at open records every
    /// batch at the depth: the measured control, not a contract.
    ramp: Cell<u32>,
    batch_ramp: Cell<bool>,
    /// Uploads are recorded into the batch (2026-09-10) when this is on, the
    /// asynchronous flush is on and the timestamps instrument off. OFF by
    /// default in the crate: the eager upload (its own fenced submission after
    /// an `Upload` flush) is the accounting law the crate's tests pin, and a
    /// recorded upload opens a batch and holds a staging pair in it. On under
    /// `ACK_DEFERRED_UPLOAD=1` at open, which the torch adapter sets before it
    /// opens, or `set_deferred_upload(true)`.
    deferred_upload: Cell<bool>,
    /// Monotonic for the life of the context: the sequence number of the next
    /// recorded item, so a flush failure can name the items it covered.
    next_item_seq: Cell<u64>,
    /// `batch::MAX_RECORDED_ITEMS`, or `ACK_MAX_RECORDED_ITEMS`. 1 is the
    /// eager arm.
    max_recorded_items: Cell<u32>,
    /// `batch::MAX_RETIRED_BYTES`, or `ACK_MAX_RETIRED_BYTES`.
    max_retired_bytes: Cell<u64>,
    /// `batch::SETS_PER_KERNEL_POOL`, or `ACK_SETS_PER_KERNEL_POOL`; read once
    /// at open, so every kernel of one context has the same arena size.
    sets_per_kernel_pool: u32,
    /// Freed buffers' Vulkan pairs, held for re-use instead of destroyed,
    /// keyed by the size `raw_buffer` was called with (`bytes.next_multiple_of(4)`).
    ///
    /// WHAT PUTS A PAIR IN HERE IS THE WHOLE SAFETY ARGUMENT. Only three
    /// places do, and every one of them is a point at which no command that
    /// names the buffer can still run: `release_retired` (after the flush's
    /// fence wait returned, or after an established device loss),
    /// `abandon_batch` (the commands naming it are being dropped unsubmitted)
    /// and `free_buffer`'s no-recorder arm (nothing is recorded and, since
    /// every submission this crate makes is fenced and waited inside the call
    /// that makes it, nothing is in flight). A free that RETIRES a buffer into
    /// an open batch does NOT reach the pool -- it reaches this map later,
    /// through `release_retired`, which is exactly where the retirement's own
    /// destroy used to happen. Handing a pair back any earlier is a wrong
    /// answer and not a leak, and `PoolMode::Immediate` is the sabotage that
    /// shows it.
    pool: RefCell<HashMap<u64, Vec<PooledBuffer>>>,
    /// Bytes the pool holds, against `pool_max_bytes`.
    pool_bytes: Cell<u64>,
    /// Ceiling on `pool_bytes`. A return that would cross it destroys the pair
    /// instead of holding it (`pool_declined`), which is O(1) and bounded; the
    /// registered pass allocates only 30 distinct sizes, so the pool reaches a
    /// steady state rather than fragmenting. `ACK_BUFFER_POOL_BYTES`.
    pool_max_bytes: Cell<u64>,
    /// Whether the pool is on. `ACK_BUFFER_POOL=0` at open, or
    /// `set_buffer_pool_enabled` per context -- the setter exists because the
    /// leak and retirement suites need the un-pooled arm on their OWN context
    /// while other tests run in the same process, which an environment
    /// variable cannot give them.
    pool_enabled: Cell<bool>,
    /// A `Context::buffer` served from the pool.
    pool_hits: Cell<u64>,
    /// A `Context::buffer` that had to call `raw_buffer`.
    pool_misses: Cell<u64>,
    /// A freed pair the pool took back.
    pool_returns: Cell<u64>,
    /// A freed pair destroyed because taking it back would cross the ceiling.
    pool_declined: Cell<u64>,
    /// Pairs destroyed when the pool was drained (teardown, or `clear_pool`).
    pool_drained: Cell<u64>,
    /// Allocation failures that drained the pool and retried.
    pool_reclaims: Cell<u64>,

    // ---- dispatch instrumentation (see `DispatchStats`) -----------------
    /// Items `Context::dispatch` recorded over this context's life.
    disp_items: Cell<u64>,
    /// `vkCmdDispatch` commands recorded. NOT the same number as `disp_items`:
    /// an item with `repeats > 1` records one command per repeat, which is why
    /// a dispatch count taken above this layer (one per C entry) and one taken
    /// here disagree, and why both are reported.
    disp_commands: Cell<u64>,
    /// Pipeline barriers actually recorded: R1's inter-item/repeat barriers
    /// and, when timing is enabled, one origin dependency per batch.
    /// It counts what reached the command buffer, so the `batch-sabotage`
    /// feature's `BarrierMode::None` makes it disagree with
    /// `recorded_barriers()`, which counts what the rule demanded.
    disp_barriers: Cell<u64>,
    /// `ACK_DISPATCH_TIMESTAMPS` at open, or `set_dispatch_timestamps` per
    /// context. OFF by default: the instrument writes a timestamp per item and
    /// reads them back at every flush, and an instrument that is always on is
    /// a change to the thing it measures.
    ///
    /// A `Cell` and not a `bool` for the same reason `pool_enabled` is one:
    /// the falsifier needs the timed and the untimed arm on its OWN context
    /// while other tests run in the same process, and an environment variable
    /// read at open cannot give it that.
    ts_enabled: Cell<bool>,
    /// This context's timestamp query pool, created at the first batch opened
    /// while `ts_enabled` and destroyed at teardown. Context-owned rather than
    /// batch-owned ON PURPOSE: a per-batch pool would have to be released or
    /// leaked on every one of the flush's failure arms, so an instrument that
    /// is off by default would still have rewritten the failure accounting the
    /// crate's poison ladder rests on. Reuse is safe because a flush waits its
    /// own fence before returning, so the batch that wrote these queries has
    /// completed before the next batch records its reset.
    ts_pool: Cell<vk::QueryPool>,
    /// Queries in `ts_pool`: `max_recorded_items + 1` at the moment it was
    /// created (one origin stamp plus one per item).
    ts_capacity: Cell<u32>,
    /// Flushes whose stamps were read back and accumulated.
    ts_flushes: Cell<u64>,
    /// Items covered by those readbacks; the denominator of the per-item
    /// figures, and NOT `disp_items` (a batch abandoned or lost is not timed).
    ts_items: Cell<u64>,
    /// Device-elapsed nanoseconds summed over the timed items.
    ts_span_ns: Cell<u64>,
    /// The shortest and longest single item interval seen. `u64::MAX` for the
    /// minimum means nothing has been timed.
    ts_min_item_ns: Cell<u64>,
    ts_max_item_ns: Cell<u64>,
    /// Flushes that carried stamps and whose span was NOT accumulated: the
    /// submission, wait or readback failed, a wrap was ambiguous, or the batch
    /// exceeded the query capacity. Reported explicitly so a partial figure
    /// cannot read as a complete one.
    ts_unresolved: Cell<u64>,
    #[cfg(feature = "pool-sabotage")]
    pool_mode: Cell<PoolMode>,
    #[cfg(feature = "batch-sabotage")]
    barrier_mode: Cell<BarrierMode>,
    #[cfg(feature = "batch-sabotage")]
    descriptor_mode: Cell<DescriptorMode>,
    #[cfg(feature = "fault-injection")]
    fault: RefCell<Vec<Fault>>,
}

/// A buffer's Vulkan objects, freed by the caller while recorded work may
/// still reference them, waiting for the flush that makes destroying them safe.
struct RetiredBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// The size `raw_buffer` was called with, so the pool can key on it
    /// without re-deriving it from a logical size the caller no longer holds.
    allocated: u64,
}

/// A freed buffer's Vulkan pair, held by the pool for re-use.
///
/// One `VkBuffer` per pair and one `vkAllocateMemory` per pair, unchanged from
/// what `raw_buffer` produces: this is a free list, NOT a suballocator. That
/// distinction is load-bearing and not a style choice. Seven refusals in
/// `ffi.rs` and one in `pointwise_ops.rs` read `Buffer::id` inequality as
/// memory-disjointness, so two live ids on one Vulkan allocation would make
/// all eight go silent at once, in the direction of admitting the dispatch.
/// A free list keeps the invariant those refusals rest on -- AT MOST ONE LIVE
/// `Buffer::id` PER PHYSICAL ALLOCATION -- true by construction, because a
/// pair is in this list exactly when no `Buffer` names it.
struct PooledBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocated: u64,
}

/// A kernel's four Vulkan objects, destroyed by the caller while recorded work
/// may still name them, waiting for the flush that makes destroying them safe.
///
/// A recorded dispatch holds `cmd_bind_pipeline(kernel.pipeline)` and
/// `cmd_bind_descriptor_sets(kernel.layout, set)` where `set` came from this
/// pool, so destroying any of the four before the submission moves that
/// command buffer to the invalid state and the flush's `vkQueueSubmit` is then
/// undefined. This is the kernel's half of the retirement `RetiredBuffer` is
/// the buffer's half of, and it exists for the same reason.
struct RetiredKernel {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
}

/// One open deferred batch: the Vulkan handles it records into, and the plan
/// (`batch::BatchPlan`) that makes every decision about it.
///
/// The command buffer and the fence are allocated when the batch opens and
/// released by the flush through `release_submission`, exactly as a one-shot
/// submission's are. That is deliberate: keeping them alive across flushes
/// would save one allocation per FLUSH (not per dispatch, which is the cost
/// this increment removes) and would change what `live_objects()` counts on a
/// context that has recorded anything, which is pinned by `tests/leaks.rs`.
struct Recorder {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    plan: batch::BatchPlan,
    /// Destroyed after this batch's fence signals; leaked and counted if the
    /// flush poisons the context with the device state unknown.
    retired: Vec<RetiredBuffer>,
    /// The same, for kernels destroyed while this batch was open. Kept in its
    /// own list because a kernel is four objects and a buffer is two, and the
    /// leak accounting counts objects, not entries.
    retired_kernels: Vec<RetiredKernel>,
    /// Timestamps this batch wrote into the context's query pool: index 0 is
    /// the batch's origin stamp, index i the completion of its ith item. 0
    /// when the instrument is off, which is the default.
    stamps: u32,
    /// This batch recorded more items than the query pool has queries, so its
    /// span covers only a prefix. Counted as unresolved at the flush rather
    /// than accumulated short: a partial span that reads as a whole one is a
    /// wrong number, not a missing one.
    ts_overflow: bool,
    /// Staging pairs of the uploads recorded into this batch (2026-09-10):
    /// host-visible memory, destroyed when the batch's fence signals, never
    /// pooled.
    retired_staging: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// A recorded upload's copy is the last command: the next item takes a
    /// barrier whether or not it is the batch's first dispatch.
    pending_transfer: bool,
}

/// One deferred batch that has been submitted and not finished: what `finish`
/// needs to wait on its fence, read its stamps, destroy what it retired and
/// free its two handles. A batch stays here between a non-completing flush
/// (`BatchFull`, `DescriptorsExhausted`) and the next flush, which is the
/// window in which the host records the next batch while the device runs
/// this one (asynchronous flushes, 2026-09-10).
struct Submitted {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    what: String,
    retired: Vec<RetiredBuffer>,
    retired_kernels: Vec<RetiredKernel>,
    stamps: u32,
    ts_overflow: bool,
    timing_started: std::time::Instant,
    /// The descriptor bank its kernels' sets came from; the next batch takes the other.
    bank: usize,
    retired_staging: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

/// One Vulkan object creation a `Context` method performs; the unit the
/// creation wrappers count and fault injection can fail (ACK-L0-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateStep {
    Buffer,
    /// Not a creation: a stand-in for the memory-type lookup refusing after
    /// the buffer exists (the real refusal is `Unsupported`; the injected one
    /// reports the given `vk::Result`, the cleanup path is the same).
    MemoryType,
    Memory,
    BindMemory,
    ShaderModule,
    DescriptorSetLayout,
    PipelineLayout,
    Pipeline,
    DescriptorPool,
    DescriptorSet,
    Fence,
    CommandBuffer,
    QueryPool,
}

/// When a freed buffer's pair becomes available for re-use (`pool-sabotage`
/// feature only; see `Context::set_pool_mode`).
///
/// `Immediate` is a SABOTAGE, not a mode anybody should run: it hands a pair
/// back the instant the caller frees it, EVEN WHILE a recorded and unflushed
/// dispatch still names that `VkBuffer` through a descriptor set. The next
/// `Context::buffer` of the same size then returns that pair to a caller who
/// believes it holds a fresh allocation, and the pending dispatch writes into
/// it when the batch is finally submitted. The bytes the new owner reads are
/// simply someone else's, produced with every counter reporting success.
///
/// It exists so a test can show that the retirement boundary is load-bearing
/// on the card in front of it, rather than asserting a rule that would pass
/// whether or not the code respected it.
#[cfg(feature = "pool-sabotage")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolMode {
    /// A pair is re-usable only once no recorded or submitted command can name
    /// it: at `release_retired`, at `abandon_batch`, or at a free with nothing
    /// recorded. This is the shipped behaviour.
    Retired,
    /// SABOTAGE: re-usable the instant it is freed.
    Immediate,
}

/// Whether the barrier R1 demands between adjacent items of a batch is
/// recorded (`batch-sabotage` feature only; see `Context::set_barrier_mode`).
///
/// `None` is a SABOTAGE, not a mode anybody should run: it removes the only
/// thing ordering one recorded dispatch against the next, so a dependent
/// chain may read what has not been written yet. It exists so a test can show
/// the barrier is load-bearing on the card in front of it, instead of
/// asserting a rule that would pass whether or not the code recorded anything.
#[cfg(feature = "batch-sabotage")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierMode {
    All,
    None,
}

/// Whether a recorded dispatch takes its own descriptor set
/// (`batch-sabotage` feature only; see `Context::set_descriptor_mode`).
///
/// `SingleSet` is a SABOTAGE: it reverts to the one set per kernel the runtime
/// had before deferred submission, so two dispatches of one kernel recorded
/// into one batch share a set and the first executes against the second's
/// buffers. It exists so a test can show the arena is load-bearing.
#[cfg(feature = "batch-sabotage")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorMode {
    Arena,
    SingleSet,
}

/// A result to substitute for a real one, after the real call has run
/// (`fault-injection` feature only; see `Context::inject_fault`).
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, Copy)]
pub enum Fault {
    /// The next `wait_for_fences` reports this result.
    WaitFails(vk::Result),
    /// The next `device_wait_idle` reports this result.
    IdleFails(vk::Result),
    /// The next creation of this kind reports this result; the object the
    /// real call made is destroyed by the wrapper first, so the caller's
    /// cleanup path runs with nothing of the injection's own left behind.
    CreateFails(CreateStep, vk::Result),
}

impl Context {
    /// Open the first discrete GPU that advertises VK_KHR_cooperative_matrix
    /// and VK_KHR_shader_bfloat16 (falling back to any device with cooperative
    /// matrices, then to any compute-capable device), with the features the
    /// kernels need enabled.
    pub fn open() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| AckError::Load(format!("{e}")))?;
        let app_name = c"alelyon-compute-kit";
        let app = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(1)
            .engine_name(app_name)
            .engine_version(1)
            .api_version(vk::make_api_version(0, 1, 4, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;
        // ACK-L0-05: any failure after this point would otherwise leak the
        // instance (and the device once created); the guard destroys them
        let guard = instance.clone();
        // the entry is the loader library: ash forbids any Vulkan call after
        // it is dropped, so `open_on` borrows it and the destroy below runs
        // while it is still alive (a review found the old by-value move
        // unloaded the loader before the guard's destroy on every failure)
        let opened = Self::open_on(&entry, instance);
        match opened {
            Ok(ctx) => Ok(ctx),
            Err(e) => {
                unsafe { guard.destroy_instance(None) };
                drop(entry);
                Err(e)
            }
        }
    }

    fn open_on(entry: &ash::Entry, instance: ash::Instance) -> Result<Self> {
        let physicals = unsafe { instance.enumerate_physical_devices() }?;
        if physicals.is_empty() {
            return Err(AckError::NoDevice("no Vulkan physical devices".into()));
        }
        let coop_ext = ash::khr::cooperative_matrix::Instance::new(entry, &instance);

        // Rank: discrete + coopmat + bf16 first.
        let mut best: Option<(i32, vk::PhysicalDevice)> = None;
        for &pd in &physicals {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let exts = unsafe { instance.enumerate_device_extension_properties(pd) }?;
            let has = |name: &str| {
                exts.iter().any(|e| {
                    e.extension_name_as_c_str()
                        .map(|c| c.to_string_lossy() == name)
                        .unwrap_or(false)
                })
            };
            let mut score = 0;
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                score += 100;
            }
            if has("VK_KHR_cooperative_matrix") {
                score += 10;
            }
            if has("VK_KHR_shader_bfloat16") {
                score += 1;
            }
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if !families
                .iter()
                .any(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                continue;
            }
            if best.map(|(s, _)| score > s).unwrap_or(true) {
                best = Some((score, pd));
            }
        }
        let (_, physical) =
            best.ok_or_else(|| AckError::NoDevice("no compute-capable device".into()))?;

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let exts = unsafe { instance.enumerate_device_extension_properties(physical) }?;
        let has_ext = |name: &str| {
            exts.iter().any(|e| {
                e.extension_name_as_c_str()
                    .map(|c| c.to_string_lossy() == name)
                    .unwrap_or(false)
            })
        };
        let has_coopmat = has_ext("VK_KHR_cooperative_matrix");
        let has_bf16 = has_ext("VK_KHR_shader_bfloat16");

        // Query features so the report says what the driver said, not what we asked for.
        let mut bf16_q = ShaderBfloat16FeaturesKHR::default();
        let mut coop_q = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        if has_bf16 {
            coop_q.p_next = (&mut bf16_q as *mut ShaderBfloat16FeaturesKHR).cast::<c_void>();
        }
        let mut v12_q = vk::PhysicalDeviceVulkan12Features::default();
        let mut v11_q = vk::PhysicalDeviceVulkan11Features::default();
        let mut f2_q = vk::PhysicalDeviceFeatures2::default();
        if has_coopmat {
            f2_q = f2_q.push_next(&mut coop_q);
        }
        f2_q = f2_q.push_next(&mut v12_q).push_next(&mut v11_q);
        unsafe { instance.get_physical_device_features2(physical, &mut f2_q) };
        // read before the chained borrows of f2_q are used again below
        let shader_int16 = f2_q.features.shader_int16 == vk::TRUE;

        let mut subgroup_q = vk::PhysicalDeviceSubgroupProperties::default();
        let mut driver_q = vk::PhysicalDeviceDriverProperties::default();
        let mut id_q = vk::PhysicalDeviceIDProperties::default();
        let mut p2_q = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_q)
            .push_next(&mut driver_q)
            .push_next(&mut id_q);
        unsafe { instance.get_physical_device_properties2(physical, &mut p2_q) };

        let coopmat_shapes = if has_coopmat {
            let rows =
                unsafe { coop_ext.get_physical_device_cooperative_matrix_properties(physical) }?;
            rows.iter()
                .map(|r| CoopmatShape {
                    m: r.m_size,
                    n: r.n_size,
                    k: r.k_size,
                    a_type: component_type_name(r.a_type),
                    b_type: component_type_name(r.b_type),
                    c_type: component_type_name(r.c_type),
                    result_type: component_type_name(r.result_type),
                    scope: scope_name(r.scope),
                    saturating: r.saturating_accumulation == vk::TRUE,
                })
                .collect()
        } else {
            Vec::new()
        };

        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_local_bytes = memory
            .memory_heaps
            .iter()
            .take(memory.memory_heap_count as usize)
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);

        let api = props.api_version;
        let report = DeviceReport {
            device_name: props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default(),
            api_version: format!(
                "{}.{}.{}",
                vk::api_version_major(api),
                vk::api_version_minor(api),
                vk::api_version_patch(api)
            ),
            driver_version_raw: props.driver_version,
            driver_name: driver_q
                .driver_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default(),
            driver_info: driver_q
                .driver_info_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default(),
            subgroup_size: subgroup_q.subgroup_size,
            timestamp_period_ns: props.limits.timestamp_period,
            bf16_type: bf16_q.shader_bfloat16_type == vk::TRUE,
            bf16_dot_product: bf16_q.shader_bfloat16_dot_product == vk::TRUE,
            bf16_cooperative_matrix: bf16_q.shader_bfloat16_cooperative_matrix == vk::TRUE,
            cooperative_matrix: coop_q.cooperative_matrix == vk::TRUE,
            shader_int16,
            coopmat_shapes,
            device_local_bytes,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type: props.device_type.as_raw() as u32,
            api_version_raw: api,
            driver_id: driver_q.driver_id.as_raw(),
            device_uuid: id_q.device_uuid,
            driver_uuid: id_q.driver_uuid,
            max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
            max_workgroup_size: props.limits.max_compute_work_group_size,
        };

        // Queue family: prefer a compute-only family (async compute), else any compute.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let mut queue_family = None;
        for (i, f) in families.iter().enumerate() {
            let compute = f.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let graphics = f.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            if compute && !graphics {
                queue_family = Some(i as u32);
                break;
            }
        }
        if queue_family.is_none() {
            queue_family = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);
        }
        let queue_family =
            queue_family.ok_or_else(|| AckError::NoDevice("no compute queue family".into()))?;
        let timestamp_valid_bits = families[queue_family as usize].timestamp_valid_bits;
        if timestamp_valid_bits == 0 {
            return Err(AckError::Unsupported(
                "compute queue has no timestamp support".into(),
            ));
        }

        // Enable exactly the features the kernels use. The f32 -> bf16 cast
        // kernel declares the SPIR-V Int16 capability (the only module that
        // does), which is legal only with shaderInt16 enabled (the Khronos
        // validation layer reported the missing feature,
        // VUID-VkShaderModuleCreateInfo-pCode-08740): require it here. Both
        // casts declare StorageBuffer16BitAccess, which needs the Vulkan 1.1
        // feature storageBuffer16BitAccess: required the same way, so a
        // driver that lacks either is refused by name at open rather than
        // by the layer at the first kernel build.
        if !shader_int16 {
            return Err(AckError::Unsupported(
                "device lacks shaderInt16, which the kit's f32->bf16 cast kernel declares (SPIR-V Int16)".into(),
            ));
        }
        if v11_q.storage_buffer16_bit_access != vk::TRUE {
            return Err(AckError::Unsupported(
                "device lacks storageBuffer16BitAccess, which both kit cast kernels declare (SPIR-V StorageBuffer16BitAccess)".into(),
            ));
        }
        let mut bf16_on = ShaderBfloat16FeaturesKHR {
            shader_bfloat16_type: bf16_q.shader_bfloat16_type,
            shader_bfloat16_dot_product: bf16_q.shader_bfloat16_dot_product,
            shader_bfloat16_cooperative_matrix: bf16_q.shader_bfloat16_cooperative_matrix,
            ..ShaderBfloat16FeaturesKHR::default()
        };
        let mut coop_on = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(coop_q.cooperative_matrix == vk::TRUE);
        if has_bf16 {
            coop_on.p_next = (&mut bf16_on as *mut ShaderBfloat16FeaturesKHR).cast::<c_void>();
        }
        let mut v12_on = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(v12_q.shader_float16 == vk::TRUE)
            .shader_int8(v12_q.shader_int8 == vk::TRUE)
            .vulkan_memory_model(v12_q.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(v12_q.vulkan_memory_model_device_scope == vk::TRUE)
            .host_query_reset(v12_q.host_query_reset == vk::TRUE);
        let mut v11_on =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        // Subgroup size control (core 1.3): tiled kernels are written for one
        // subgroup width and ask for it explicitly rather than trusting a heuristic.
        let mut v13_q = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2_13 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13_q);
        unsafe { instance.get_physical_device_features2(physical, &mut f2_13) };
        let subgroup_size_control =
            v13_q.subgroup_size_control == vk::TRUE && v13_q.compute_full_subgroups == vk::TRUE;
        let mut v13_on = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(subgroup_size_control)
            .compute_full_subgroups(subgroup_size_control);
        let mut f2_on = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int16(true));
        if has_coopmat {
            f2_on = f2_on.push_next(&mut coop_on);
        }
        f2_on = f2_on
            .push_next(&mut v13_on)
            .push_next(&mut v12_on)
            .push_next(&mut v11_on);

        let priorities = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut ext_names: Vec<*const i8> = Vec::new();
        let coop_name = c"VK_KHR_cooperative_matrix";
        let bf16_name = c"VK_KHR_shader_bfloat16";
        if has_coopmat {
            ext_names.push(coop_name.as_ptr());
        }
        if has_bf16 {
            ext_names.push(bf16_name.as_ptr());
        }
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&ext_names)
            .push_next(&mut f2_on);
        let device = unsafe { instance.create_device(physical, &device_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(e) => {
                // the device exists and nothing else does: release it (ACK-L0-05)
                unsafe { device.destroy_device(None) };
                return Err(e.into());
            }
        };

        let device_local_bytes = report.device_local_bytes;
        Ok(Self {
            _entry: std::mem::ManuallyDrop::new(entry.clone()),
            instance,
            physical,
            device,
            queue,
            queue_family,
            command_pool,
            memory,
            report,
            subgroup_size_control,
            timestamp_valid_bits,
            max_storage_buffer_bytes: props.limits.max_storage_buffer_range as u64,
            max_workgroup_count: props.limits.max_compute_work_group_count,
            max_push_constant_bytes: props.limits.max_push_constants_size,
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
            live_buffers: RefCell::new(HashSet::new()),
            next_buffer: Cell::new(1),
            _not_sync: PhantomData,
            poisoned: RefCell::new(None),
            lost: Cell::new(false),
            leaked: Cell::new(0),
            torn_down: Cell::new(0),
            live_objects: Cell::new(0),
            recorder: RefCell::new(None),
            flushes: std::array::from_fn(|_| Cell::new(0)),
            batch_gen: Cell::new(1),
            bank: Cell::new(0),
            in_flight: RefCell::new(None),
            async_flush: Cell::new(batch::env_flag("ACK_ASYNC_FLUSH", true)),
            ramp: Cell::new(0),
            batch_ramp: Cell::new(batch::env_flag("ACK_BATCH_RAMP", true)),
            deferred_upload: Cell::new(batch::env_flag("ACK_DEFERRED_UPLOAD", false)),
            next_item_seq: Cell::new(0),
            max_recorded_items: Cell::new(batch::env_u32(
                "ACK_MAX_RECORDED_ITEMS",
                batch::MAX_RECORDED_ITEMS,
            )),
            max_retired_bytes: Cell::new(batch::env_u64(
                "ACK_MAX_RETIRED_BYTES",
                batch::MAX_RETIRED_BYTES,
            )),
            sets_per_kernel_pool: batch::env_u32(
                "ACK_SETS_PER_KERNEL_POOL",
                batch::SETS_PER_KERNEL_POOL,
            ),
            pool: RefCell::new(HashMap::new()),
            pool_bytes: Cell::new(0),
            // HALF the device-local heap, and the half is MEASURED rather
            // than chosen. A quarter (3.98 GiB on this card) was the first
            // default and it BINDS at the registered A2 geometry: the step is
            // 21.98 s there, 18.37 s at a 6 GiB ceiling and 18.16 s at 8 GiB,
            // so the pool's working set is about 5-6 GiB and a quarter-heap
            // ceiling was declining returns worth 17% of the step. Half the
            // heap clears it with room left over.
            //
            // Raising it is safe because `Context::buffer` drains the pool and
            // retries when an allocation fails, so a pool holding unowned
            // memory can never be the reason a caller cannot allocate. Without
            // that retry this constant would be a safety margin and would have
            // to stay low. UNMEASURED at any other geometry or on any other
            // card; `ACK_BUFFER_POOL_BYTES` overrides it.
            pool_max_bytes: Cell::new(batch::env_u64(
                "ACK_BUFFER_POOL_BYTES",
                (device_local_bytes / 2).max(64 << 20),
            )),
            pool_enabled: Cell::new(batch::env_flag("ACK_BUFFER_POOL", true)),
            pool_hits: Cell::new(0),
            pool_misses: Cell::new(0),
            pool_returns: Cell::new(0),
            pool_declined: Cell::new(0),
            pool_drained: Cell::new(0),
            pool_reclaims: Cell::new(0),
            disp_items: Cell::new(0),
            disp_commands: Cell::new(0),
            disp_barriers: Cell::new(0),
            ts_enabled: Cell::new(batch::env_flag("ACK_DISPATCH_TIMESTAMPS", false)),
            ts_pool: Cell::new(vk::QueryPool::null()),
            ts_capacity: Cell::new(0),
            ts_flushes: Cell::new(0),
            ts_items: Cell::new(0),
            ts_span_ns: Cell::new(0),
            ts_min_item_ns: Cell::new(u64::MAX),
            ts_max_item_ns: Cell::new(0),
            ts_unresolved: Cell::new(0),
            #[cfg(feature = "pool-sabotage")]
            pool_mode: Cell::new(PoolMode::Retired),
            #[cfg(feature = "batch-sabotage")]
            barrier_mode: Cell::new(BarrierMode::All),
            #[cfg(feature = "batch-sabotage")]
            descriptor_mode: Cell::new(DescriptorMode::Arena),
            #[cfg(feature = "fault-injection")]
            fault: RefCell::new(Vec::new()),
        })
    }

    /// Stop recording the barrier R1 demands between adjacent items of a
    /// batch. A SABOTAGE (`batch-sabotage` feature); see `BarrierMode`.
    #[cfg(feature = "batch-sabotage")]
    pub fn set_barrier_mode(&self, mode: BarrierMode) {
        self.barrier_mode.set(mode);
    }

    /// Revert to one descriptor set per kernel. A SABOTAGE
    /// (`batch-sabotage` feature); see `DescriptorMode`.
    #[cfg(feature = "batch-sabotage")]
    pub fn set_descriptor_mode(&self, mode: DescriptorMode) {
        self.descriptor_mode.set(mode);
    }

    #[cfg(feature = "batch-sabotage")]
    fn barriers_enabled(&self) -> bool {
        self.barrier_mode.get() == BarrierMode::All
    }

    #[cfg(not(feature = "batch-sabotage"))]
    fn barriers_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "batch-sabotage")]
    fn arena_enabled(&self) -> bool {
        self.descriptor_mode.get() == DescriptorMode::Arena
    }

    #[cfg(not(feature = "batch-sabotage"))]
    fn arena_enabled(&self) -> bool {
        true
    }

    /// Whether a freed pair may be re-used the instant it is freed. A
    /// SABOTAGE (`pool-sabotage` feature); see `PoolMode`.
    #[cfg(feature = "pool-sabotage")]
    pub fn set_pool_mode(&self, mode: PoolMode) {
        self.pool_mode.set(mode);
    }

    #[cfg(feature = "pool-sabotage")]
    fn pool_defers_to_retirement(&self) -> bool {
        self.pool_mode.get() == PoolMode::Retired
    }

    #[cfg(not(feature = "pool-sabotage"))]
    fn pool_defers_to_retirement(&self) -> bool {
        true
    }

    /// Turn the buffer pool on or off for THIS context. Off, `Context::buffer`
    /// allocates and `destroy_buffer` destroys exactly as they did before the
    /// pool existed, which is the arm the leak and retirement suites assert
    /// their original object deltas on.
    ///
    /// Turning it off DRAINS what it holds, so `live_objects()` is back to the
    /// un-pooled quantity immediately rather than at the next teardown.
    pub fn set_buffer_pool_enabled(&self, on: bool) {
        self.pool_enabled.set(on);
        if !on {
            self.clear_pool();
        }
    }

    /// Whether the buffer pool is on for this context.
    pub fn buffer_pool_enabled(&self) -> bool {
        self.pool_enabled.get()
    }

    /// Ceiling on the bytes the pool holds; a return that would cross it
    /// destroys the pair instead (`pool_declined`).
    pub fn set_buffer_pool_max_bytes(&self, bytes: u64) {
        self.pool_max_bytes.set(bytes);
        self.evict_to_ceiling();
    }

    /// `Context::buffer` calls served from the pool, calls that had to
    /// allocate, pairs the pool took back, pairs destroyed at the ceiling, and
    /// pairs destroyed by a drain.
    ///
    /// ASSERT THESE ALONGSIDE THE ARITHMETIC. A correctness probe cannot tell
    /// you a buffer was recycled rather than freshly allocated: a pool that
    /// silently allocated every time would pass every value assertion in the
    /// suite. The hit counter is the bracket that makes those assertions
    /// non-vacuous, in the same way the fallback counter -- and nothing else --
    /// caught a sabotaged bound this morning.
    pub fn pool_stats(&self) -> PoolStats {
        PoolStats {
            hits: self.pool_hits.get(),
            misses: self.pool_misses.get(),
            returns: self.pool_returns.get(),
            declined: self.pool_declined.get(),
            drained: self.pool_drained.get(),
            reclaims: self.pool_reclaims.get(),
            held_objects: self.pooled_objects(),
            held_bytes: self.pool_bytes.get(),
        }
    }

    /// Vulkan objects the pool holds (two per pair). They are LIVE objects and
    /// `live_objects()` counts them, because they are: the pool is holding
    /// them, not leaking them. A leak assertion that wants the un-pooled
    /// quantity subtracts this, or runs on a context with the pool off.
    pub fn pooled_objects(&self) -> i64 {
        self.pool
            .borrow()
            .values()
            .map(|v| v.len() as i64 * 2)
            .sum()
    }

    /// Take a pair of the requested allocated size, if the pool holds one.
    fn take_pooled(&self, allocated: u64) -> Option<PooledBuffer> {
        if !self.pool_enabled.get() {
            return None;
        }
        let mut pool = self.pool.borrow_mut();
        let entry = pool.get_mut(&allocated)?;
        let p = entry.pop()?;
        if entry.is_empty() {
            pool.remove(&allocated);
        }
        self.pool_bytes.set(self.pool_bytes.get() - p.allocated);
        Some(p)
    }

    /// Take a freed pair back for re-use, or destroy it.
    ///
    /// THE CALLER OWNS THE SAFETY ARGUMENT. Every call site must be a point at
    /// which no recorded-but-unflushed and no submitted-but-unwaited command
    /// names this buffer; see the `pool` field for the enumeration of the
    /// three that qualify. This function cannot check that and does not try.
    fn recycle_or_destroy(&self, buffer: vk::Buffer, memory: vk::DeviceMemory, allocated: u64) {
        if self.pool_enabled.get() && self.pool_bytes.get() + allocated <= self.pool_max_bytes.get()
        {
            self.pool
                .borrow_mut()
                .entry(allocated)
                .or_default()
                .push(PooledBuffer {
                    buffer,
                    memory,
                    allocated,
                });
            self.pool_bytes.set(self.pool_bytes.get() + allocated);
            self.pool_returns.set(self.pool_returns.get() + 1);
            return;
        }
        if self.pool_enabled.get() {
            self.pool_declined.set(self.pool_declined.get() + 1);
        }
        self.vk_destroy_buffer(buffer);
        self.vk_free_memory(memory);
    }

    /// Destroy every pair the pool holds. Called by teardown, by
    /// `set_buffer_pool_enabled(false)` and when the ceiling is lowered.
    fn clear_pool(&self) {
        let held: Vec<PooledBuffer> = self
            .pool
            .borrow_mut()
            .drain()
            .flat_map(|(_, v)| v)
            .collect();
        for p in held {
            self.vk_destroy_buffer(p.buffer);
            self.vk_free_memory(p.memory);
            self.pool_drained.set(self.pool_drained.get() + 1);
        }
        self.pool_bytes.set(0);
    }

    /// Destroy pairs until the pool is inside its ceiling.
    fn evict_to_ceiling(&self) {
        while self.pool_bytes.get() > self.pool_max_bytes.get() {
            let taken = {
                let mut pool = self.pool.borrow_mut();
                let key = *match pool.keys().next() {
                    Some(k) => k,
                    None => break,
                };
                let entry = pool.get_mut(&key).expect("key just read");
                let p = entry.pop();
                if entry.is_empty() {
                    pool.remove(&key);
                }
                p
            };
            match taken {
                Some(p) => {
                    self.pool_bytes
                        .set(self.pool_bytes.get().saturating_sub(p.allocated));
                    self.vk_destroy_buffer(p.buffer);
                    self.vk_free_memory(p.memory);
                    self.pool_drained.set(self.pool_drained.get() + 1);
                }
                None => break,
            }
        }
    }

    /// Refuse by name once the device state is unknown (ACK-L0-03).
    fn check_live(&self) -> Result<()> {
        match self.poisoned.borrow().as_ref() {
            Some(why) => Err(AckError::Poisoned(why.clone())),
            None => Ok(()),
        }
    }

    fn poison(&self, why: String, lost: bool) {
        *self.poisoned.borrow_mut() = Some(why);
        if lost {
            self.lost.set(true);
        }
    }

    /// Whether a failed wait left this context unusable.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.borrow().is_some()
    }

    /// Objects left allocated on purpose because the context is poisoned.
    pub fn leaked_on_poison(&self) -> u64 {
        self.leaked.get()
    }

    /// Submissions whose fence and command buffer were released after their
    /// completion or after an established device loss.
    pub fn submissions_torn_down(&self) -> u64 {
        self.torn_down.get()
    }

    /// Destroy objects a public method created around a submission when the
    /// device is quiesced or lost, or leave them allocated and count them when
    /// the context is poisoned with the device state unknown. The one place
    /// that decides destroy-or-leak for caller-owned objects.
    fn release_or_leak(&self, objects: u64, destroy: impl FnOnce()) {
        if self.is_poisoned() && !self.lost.get() {
            self.leaked.set(self.leaked.get() + objects);
        } else {
            destroy();
        }
    }

    /// Release a submission's fence and command buffer and count the release
    /// in the same place, so the count cannot say "released" when nothing was.
    fn release_submission(&self, fence: vk::Fence, free_cb: impl FnOnce()) {
        self.vk_destroy_fence(fence);
        free_cb();
        self.torn_down.set(self.torn_down.get() + 1);
    }

    // ---- deferred submission (see `batch`) ------------------------------

    /// Flushes so far, indexed as `batch::ALL_FLUSH_REASONS` is.
    ///
    /// Incremented inside `flush` and nowhere else, once for every flush of a
    /// NON-EMPTY batch -- including one that then fails before its submission.
    /// The increment sits after the empty-batch early return and before
    /// `end_command_buffer`, so what it counts is "a batch was taken and
    /// committed to this flush", not "a submission reached the queue": a flush
    /// of an empty batch submits nothing and is not counted, and a flush that
    /// takes a batch and then fails at `end_command_buffer`, `reset_fences` or
    /// `queue_submit` IS counted, even though its own error says nothing was
    /// submitted.
    ///
    /// That is the useful reading for what this counter exists for -- a flush
    /// point that stops firing becomes a failing test rather than a stale
    /// sentence in a document -- and the direction is safe for it: this can
    /// only over-count, and only when the driver is already failing, so it
    /// cannot hide a flush point that went quiet. It is NOT a count of
    /// successful submissions and must not be read as one.
    /// Batches submitted and not yet finished: 0 or 1. A non-completing flush
    /// (`BatchFull`, `DescriptorsExhausted`, with the timestamps instrument
    /// off) leaves its batch here; the next flush finishes it.
    pub fn batches_in_flight(&self) -> u32 {
        u32::from(self.in_flight.borrow().is_some())
    }

    pub fn flushes_by_reason(&self) -> [u64; batch::FLUSH_REASON_COUNT] {
        std::array::from_fn(|i| self.flushes[i].get())
    }

    /// Every flush of this context, whatever the reason.
    pub fn flushes(&self) -> u64 {
        self.flushes.iter().map(|c| c.get()).sum()
    }

    /// Dispatches recorded into the open batch and not yet submitted.
    pub fn recorded_items(&self) -> u32 {
        self.recorder
            .borrow()
            .as_ref()
            .map_or(0, |r| r.plan.items())
    }

    /// Barriers R1 demands for the open batch. The invariant is arithmetic:
    /// this is `recorded_items() - 1`, or 0 for an empty batch.
    ///
    /// It counts what the rule demanded, not what reached the command buffer:
    /// with the `batch-sabotage` feature's `BarrierMode::None` the recorder
    /// demands them and records none, which is exactly what that sabotage is
    /// for. Without the feature the two are the same number.
    pub fn recorded_barriers(&self) -> u32 {
        self.recorder
            .borrow()
            .as_ref()
            .map_or(0, |r| r.plan.barriers())
    }

    /// Items recorded into one batch before it is submitted; 1 is the eager
    /// arm (see `batch::MAX_RECORDED_ITEMS`). Set from `ACK_MAX_RECORDED_ITEMS`
    /// at open; this setter exists so a measurement can alternate the two arms
    /// round by round inside one session rather than across two.
    /// Whether the batches after a completing flush ramp up from
    /// `batch::RAMP_FIRST` items (the default when the asynchronous flush is
    /// on and the timestamps instrument off), or every batch is the depth.
    pub fn batch_ramp(&self) -> bool {
        self.batch_ramp.get()
    }

    /// Turn the ramp on or off for this context, as `ACK_BATCH_RAMP` does at
    /// open: for a test whose subject is a batch's own limit, and for the
    /// measured control.
    pub fn set_batch_ramp(&self, on: bool) {
        self.batch_ramp.set(on);
    }

    /// Whether an upload is recorded into the batch (`ACK_DEFERRED_UPLOAD=1` at
    /// open or `set_deferred_upload(true)`, with the asynchronous flush on and
    /// the timestamps instrument off) or is its own fenced submission after a
    /// flush, the crate's default.
    pub fn deferred_upload(&self) -> bool {
        self.deferred_upload.get() && self.async_flush.get() && !self.ts_enabled.get()
    }

    /// Turn recorded uploads on or off for this context, as
    /// `ACK_DEFERRED_UPLOAD` does at open: for the test of the recorded path,
    /// and for a caller that wants it without the environment.
    pub fn set_deferred_upload(&self, on: bool) {
        self.deferred_upload.set(on);
    }

    pub fn max_recorded_items(&self) -> u32 {
        self.max_recorded_items.get()
    }

    /// Change the batch depth. Takes effect at the next recorded item; the
    /// open batch, if any, keeps the budget it was opened with.
    pub fn set_max_recorded_items(&self, items: u32) {
        self.max_recorded_items.set(items.max(1));
    }

    /// Device bytes retired into one batch before it is flushed (see
    /// `batch::MAX_RETIRED_BYTES`). Set from `ACK_MAX_RETIRED_BYTES` at open.
    pub fn max_retired_bytes(&self) -> u64 {
        self.max_retired_bytes.get()
    }

    /// Change the retirement budget. Takes effect at the next batch opened;
    /// the open batch, if any, keeps the budget it was opened with.
    pub fn set_max_retired_bytes(&self, bytes: u64) {
        self.max_retired_bytes.set(bytes);
    }

    /// Descriptor sets pre-allocated per kernel (see
    /// `batch::SETS_PER_KERNEL_POOL`).
    pub fn sets_per_kernel_pool(&self) -> u32 {
        self.sets_per_kernel_pool
    }

    /// Submit the recorded batch and wait for it, with nothing recorded
    /// afterwards. A caller that wants recorded work to have run without
    /// reading anything back.
    pub fn sync(&self) -> Result<()> {
        self.flush(batch::FlushReason::ExplicitSync)
    }

    /// Open a batch: allocate its command buffer and fence and begin
    /// recording. Every failure here frees what it created and installs
    /// nothing, so `recorder` is `Some` only for a batch that can be recorded
    /// into (ACK-L0-05).
    fn open_batch(&self) -> Result<()> {
        if self.recorder.borrow().is_some() {
            return Ok(());
        }
        // the batch about to open takes the bank the batch in flight is not using
        let _ = self.open_bank();
        // the instrument's query pool, if it is on and this is the first batch
        // (created before the command buffer, so a failure here frees nothing
        // and installs nothing, exactly as the two allocations below do)
        if self.ts_enabled.get() && self.ts_pool.get() == vk::QueryPool::null() {
            let capacity = self.max_recorded_items.get().saturating_add(1);
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(capacity);
            let qp = self.vk_create_query_pool(&info)?;
            self.ts_pool.set(qp);
            self.ts_capacity.set(capacity);
        }
        let cb = self.vk_allocate_command_buffer()?;
        let fence = match self.vk_create_fence() {
            Ok(f) => f,
            Err(e) => {
                self.vk_free_command_buffer(cb);
                return Err(e.into());
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { self.device.begin_command_buffer(cb, &begin) } {
            self.vk_destroy_fence(fence);
            self.vk_free_command_buffer(cb);
            return Err(e.into());
        }
        // the reset must be recorded before any write, and the origin stamp is
        // query 0: every later stamp is read against it
        let mut stamps = 0u32;
        if self.ts_enabled.get() {
            let qp = self.ts_pool.get();
            unsafe {
                self.device
                    .cmd_reset_query_pool(cb, qp, 0, self.ts_capacity.get());
                self.device
                    .cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 0);
                // A timestamp orders its write after earlier commands, but
                // does not hold later compute. Establish a batch origin that
                // precedes the first dispatch. This cost is instrument-only.
                self.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[],
                );
            }
            self.disp_barriers.set(self.disp_barriers.get() + 1);
            stamps = 1;
        }
        // THE CROSS-BATCH BARRIER (2026-09-10). Under the asynchronous flush
        // the batch before this one may still be executing when this one is
        // submitted, and Vulkan orders two submissions' commands only through
        // a barrier whose first scope reaches earlier submission order. The
        // R1 barrier before every item but the first covers every pair inside
        // a batch; this one covers the first item against the batch before.
        // It is the batch's, not an item's: the instrument's barrier counter
        // (items - 1 per batch, plus the origin barrier in timestamps mode,
        // where the flush is synchronous and this one is not recorded) does
        // not count it.
        if self.async_flush.get() && !self.ts_enabled.get() && self.barriers_enabled() {
            self.record_barrier(cb);
        }
        *self.recorder.borrow_mut() = Some(Recorder {
            cb,
            fence,
            plan: batch::BatchPlan::new(
                self.next_item_seq.get(),
                self.batch_depth(),
                self.max_retired_bytes.get(),
            ),
            retired: Vec::new(),
            retired_kernels: Vec::new(),
            stamps,
            ts_overflow: false,
            retired_staging: Vec::new(),
            pending_transfer: false,
        });
        Ok(())
    }

    /// R1: one full memory barrier, recorded immediately before every item of
    /// a batch except its first.
    ///
    /// WHAT THIS RECORDS: one `vkCmdPipelineBarrier` with one
    /// `VkMemoryBarrier`, compute and transfer on both stage masks,
    /// `SHADER_WRITE | TRANSFER_WRITE` to
    /// `SHADER_READ | SHADER_WRITE | TRANSFER_READ | TRANSFER_WRITE`, no
    /// dependency flags and no buffer or image barrier. Phase 1 records only
    /// dispatches, so the transfer bits cover no recorded item today; they are
    /// in the mask because widening them costs nothing here and a later
    /// recorded transfer would otherwise need the masks changed in a second
    /// place.
    ///
    /// WHAT IS PINNED: `batch::BatchPlan::record_with` decides WHEN this is
    /// recorded, from the declaration the dispatch carries; its tests hold
    /// both that a dependent chain still takes one between every pair and
    /// that disjoint dispatches take none.
    ///
    /// WHAT IS UNMEASURED: whether any driver would have overlapped two
    /// adjacent recorded dispatches without it, and therefore whether this
    /// barrier is load-bearing on any particular card. The falsifier in
    /// `tests/batch_flush.rs` removes it and requires divergence; when the
    /// sabotage does not manifest, that test reports UNMEASURED rather than
    /// passing. No claim is made here about Vulkan's or a driver's submission
    /// or overlap behaviour.
    fn record_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            )
        };
    }

    /// Destroy or leak a finished batch's retired buffers and kernels, through
    /// the one place that decides destroy-or-leak (`release_or_leak`).
    /// The staging pairs of a finished batch's recorded uploads: destroyed
    /// (host-visible memory is never pooled), or leaked and counted on a
    /// poisoned context whose device state is unknown.
    fn release_staging(&self, staging: Vec<(vk::Buffer, vk::DeviceMemory)>) {
        if staging.is_empty() {
            return;
        }
        let objects = staging.len() as u64 * 2;
        self.release_or_leak(objects, || {
            for (b, m) in staging {
                self.vk_destroy_buffer(b);
                self.vk_free_memory(m);
            }
        });
    }

    fn release_retired(&self, retired: Vec<RetiredBuffer>, kernels: Vec<RetiredKernel>) {
        if retired.is_empty() && kernels.is_empty() {
            return;
        }
        // two objects per buffer, four per kernel: the count is of OBJECTS,
        // which is what `leaked_on_poison()` and `live_objects()` count
        let objects = retired.len() as u64 * 2 + kernels.len() as u64 * 4;
        self.release_or_leak(objects, || {
            for r in retired {
                // THE EARLIEST PROVABLY SAFE POINT. Every submission this
                // crate makes is fenced and waited inside the call that makes
                // it, so by the time this runs the batch that could name these
                // buffers has either completed its wait, been established as
                // lost, or never been submitted. That is precisely the window
                // `RetiredBuffer` was written to guard, re-expressed so the
                // pair survives instead of being destroyed.
                self.recycle_or_destroy(r.buffer, r.memory, r.allocated);
            }
            for k in kernels {
                self.vk_destroy_descriptor_pool(k.pool);
                self.vk_destroy_pipeline(k.pipeline);
                self.vk_destroy_pipeline_layout(k.layout);
                self.vk_destroy_descriptor_set_layout(k.dsl);
            }
        });
    }

    /// Submit the recorded batch, wait for it, and release or leak what it
    /// held, preserving ACK-L0-03's ladder with "the call" widened to "the
    /// batch".
    ///
    /// A flush of an empty batch submits nothing: it moves the batch
    /// generation (so every kernel's descriptor watermark starts again, which
    /// is safe precisely because no recorded item holds a set) and returns
    /// `Ok`, uncounted.
    ///
    /// A FAILURE HERE CANNOT BE ATTRIBUTED TO ONE RECORDED OPERATION, and this
    /// code does not pretend it can. There is no per-dispatch completion
    /// status in use: `VK_EXT_device_fault` is not enabled by this context,
    /// and what it would report on any driver here is UNMEASURED. The error
    /// names the batch instead -- its reason, its item count and its sequence
    /// range -- and says plainly that every operation in it is of unknown
    /// completion. A caller that needs per-operation attribution runs with
    /// `ACK_MAX_RECORDED_ITEMS=1`, which is the eager arm and the same code
    /// path. A FAILED BATCH IS NEVER RETRIED: its operations may have
    /// partially executed and re-running a non-idempotent one would
    /// double-apply it.
    ///
    /// # ACK-L0-03's ladder, walked against `one_shot`'s, guarantee by
    /// guarantee
    ///
    /// This is new code standing in for the ladder that was built for
    /// `one_shot`. Each of that ladder's guarantees is named here with what
    /// happens to it, so nobody has to rediscover which ones survived.
    ///
    /// 1. *Before the submission succeeds, every error path frees what the
    ///    call created.* HOLDS, AND IS WIDER. `one_shot` frees its command
    ///    buffer and (once created) its fence. Here the same two are freed and
    ///    so is everything the caller retired into the batch, through
    ///    `release_retired`. Nothing was submitted, so nothing can be naming
    ///    them.
    ///
    /// 2. *A failed submission returns the caller to a usable context.*
    ///    DELIBERATELY WEAKENED, AND THE WEAKENING IS THE POINT. `one_shot`'s
    ///    pre-submission failures lose only that call's own work, which the
    ///    caller is being told about in the same breath, so the context stays
    ///    usable. A pre-submission failure here loses A WHOLE BATCH of
    ///    dispatches the caller was already told were ACCEPTED, and no
    ///    mechanism exists to re-run them (the command buffer is freed on the
    ///    way out, and a batch is never retried). A later readback would
    ///    therefore return bytes that predate work the caller believes ran, so
    ///    the context is POISONED and refuses every later call by name. The
    ///    objects are released before the poison is set, because at that point
    ///    the device is known idle with respect to this batch and destroying
    ///    is right; `release_or_leak`'s leak rule applies to what comes after.
    ///
    /// 3. *A device loss reported at any step is an established loss: release
    ///    and poison with `lost`.* HOLDS, AND IS NOW COMPLETE. `one_shot`
    ///    applies it at the fence wait but not at `vkQueueSubmit`, which may
    ///    also return `VK_ERROR_DEVICE_LOST`; that hole is not repeated here.
    ///    A `DEVICE_LOST` from `vkQueueSubmit` takes the same arm as one from
    ///    the wait: release, poison with `lost = true`, return
    ///    `AckError::Vulkan(ERROR_DEVICE_LOST)` so the caller sees the real
    ///    `VkResult` rather than prose.
    ///
    /// 4. *After a successful submission, a failed wait quiesces the device
    ///    with `device_wait_idle`; idle success releases and returns the wait
    ///    error with the context usable.* HOLDS UNCHANGED, with "the call's
    ///    objects" read as "the batch's": the device is quiesced, so the
    ///    batch's operations completed and nothing is lost. This is the one
    ///    failure arm here that does not poison, and that is correct.
    ///
    /// 5. *Any other idle error leaves the objects allocated, counts them in
    ///    `leaked_on_poison()`, and poisons without `lost`.* HOLDS, WIDENED:
    ///    the fence and command buffer (2) plus every retired object, counted
    ///    through `release_or_leak` rather than a second hand-written sum.
    ///
    /// 6. *`submissions_torn_down()` counts a release exactly when one
    ///    happened.* HOLDS, with the unit changed from one submission per
    ///    OPERATION to one per FLUSH. Both are `release_submission`, which
    ///    counts in the same place it destroys.
    ///
    /// 7. *The error names what failed.* NOW UNPROVABLE AT THE OPERATION
    ///    LEVEL, and this code says so instead of guessing: see the paragraph
    ///    above. `one_shot`'s error belongs to one operation because one
    ///    operation was submitted; a batch's does not.
    ///
    /// # What is UNMEASURED
    ///
    /// Every arm from 3 downwards needs a device that is failing in a
    /// particular way. `Fault` can substitute a fence-wait or device-idle
    /// result (arms 3, 4 and 5 are driven that way in `tests/batch_flush.rs`);
    /// there is NO hook for `end_command_buffer`, `reset_fences` or
    /// `queue_submit`, so arms 1, 2 and the `DEVICE_LOST`-at-submit half of
    /// arm 3 are reachable only from a real driver fault and are UNMEASURED.
    pub(crate) fn flush(&self, reason: batch::FlushReason) -> Result<()> {
        self.check_live()?;
        let taken = self.recorder.borrow_mut().take();
        // whether or not anything was recorded, no set is held after this
        self.batch_gen.set(self.batch_gen.get() + 1);
        let Some(rec) = taken else {
            // nothing recorded; a batch may still be in flight, and a reason that
            // promises completion must see it through
            if self.completes(reason) {
                self.reap_in_flight()?;
                self.ramp.set(0);
            }
            return Ok(());
        };
        self.flushes[batch::flush_reason_index(reason)]
            .set(self.flushes[batch::flush_reason_index(reason)].get() + 1);
        let cb = rec.cb;
        let fence = rec.fence;
        let cbs = [cb];
        let what = rec.plan.describe(reason);
        // read off the recorder before it is consumed by the retirement below
        let stamps = rec.stamps;
        let ts_overflow = rec.ts_overflow;
        // Count a stamped batch as unresolved until its complete span is read.
        // This also covers every submission/wait failure below; a later
        // healthy batch must not erase a gap in the timing history.
        if stamps > 1 {
            self.ts_unresolved.set(self.ts_unresolved.get() + 1);
        }

        // (a) nothing is submitted yet: every failure frees what the batch
        //     created, as one_shot's pre-submission paths do. A DEVICE_LOST
        //     from any of the three is an established loss and takes the same
        //     arm as one from the fence wait; anything else loses the batch
        //     without losing the device, which poisons for a different reason
        //     (see `lose_batch`).
        if let Err(e) = unsafe { self.device.end_command_buffer(cb) } {
            return Err(self.lose_batch(rec, e, &what, "end_command_buffer"));
        }
        if let Err(e) = unsafe { self.device.reset_fences(&[fence]) } {
            return Err(self.lose_batch(rec, e, &what, "reset_fences"));
        }
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        let timing_started = std::time::Instant::now();
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, &submit, fence) } {
            return Err(self.lose_batch(rec, e, &what, "queue_submit"));
        }

        let submitted = Submitted {
            cb,
            fence,
            what,
            retired: rec.retired,
            retired_kernels: rec.retired_kernels,
            stamps,
            ts_overflow,
            timing_started,
            bank: self.bank.get(),
            retired_staging: rec.retired_staging,
        };
        // (b) submitted. The batch before this one ran while this one was
        //     recorded; it is finished first, so its bank and its retired
        //     objects are free again before this one takes its place.
        if let Err(e) = self.reap_in_flight() {
            // the ladder poisoned the context; this batch's objects are leaked
            // as a poisoned context leaks them (counted), never destroyed
            // under work whose state is unknown
            self.leaked.set(
                self.leaked.get()
                    + 2
                    + 2 * submitted.retired.len() as u64
                    + 4 * submitted.retired_kernels.len() as u64
                    + 2 * submitted.retired_staging.len() as u64,
            );
            return Err(e);
        }
        if self.completes(reason) {
            self.ramp.set(0);
            return self.finish(submitted);
        }
        *self.in_flight.borrow_mut() = Some(submitted);
        self.ramp.set(self.ramp.get().saturating_add(1));
        Ok(())
    }

    /// Whether a flush for `reason` must see its batch complete before it
    /// returns: every reason that promises the host something about the
    /// device (a readback, an upload into a buffer the batch may read, an
    /// explicit sync, close, the timed dispatch, the retirement budget that
    /// wants its memory back), and every flush while the timestamps
    /// instrument is on, whose query pool is one batch wide. `BatchFull` and
    /// `DescriptorsExhausted` promise nothing and leave the batch in flight.
    fn completes(&self, reason: batch::FlushReason) -> bool {
        !self.async_flush.get()
            || self.ts_enabled.get()
            || !matches!(
                reason,
                batch::FlushReason::BatchFull | batch::FlushReason::DescriptorsExhausted
            )
    }

    /// Items the batch about to open may record before `BatchFull`: the depth,
    /// or on the ramp after a completing flush `RAMP_FIRST << ramp`, whichever
    /// is smaller. A depth at or below `RAMP_FIRST` (the eager arm's 1, the
    /// tests' 2 and 3) is never shortened.
    fn batch_depth(&self) -> u32 {
        let depth = self.max_recorded_items.get();
        if !self.batch_ramp.get() || !self.async_flush.get() || self.ts_enabled.get() {
            return depth;
        }
        let ramped = batch::RAMP_FIRST
            .checked_shl(self.ramp.get().min(16))
            .unwrap_or(u32::MAX);
        depth.min(ramped)
    }

    /// Finish the batch in flight, if any.
    fn reap_in_flight(&self) -> Result<()> {
        let taken = self.in_flight.borrow_mut().take();
        match taken {
            Some(s) => self.finish(s),
            None => Ok(()),
        }
    }

    /// The descriptor bank of the open batch, or of the batch about to open:
    /// the one the batch in flight is not using. Decided once per batch, by
    /// whichever comes first: the set a dispatch acquires or the batch it opens.
    fn open_bank(&self) -> usize {
        if self.recorder.borrow().is_some() {
            return self.bank.get();
        }
        let bank = match self.in_flight.borrow().as_ref() {
            Some(s) => 1 - s.bank,
            None => 0,
        };
        self.bank.set(bank);
        bank
    }

    /// See a submitted batch through: ACK-L0-03's ladder on its fence, with the
    /// caller-owned object set widened from the call's to the batch's; then its
    /// stamps, the objects it retired, its fence and its command buffer.
    fn finish(&self, submitted: Submitted) -> Result<()> {
        let Submitted {
            cb,
            fence,
            what,
            retired,
            retired_kernels,
            stamps,
            ts_overflow,
            timing_started,
            retired_staging,
            ..
        } = submitted;
        let free_cb = || self.vk_free_command_buffer(cb);
        // (b) submitted: ACK-L0-03's ladder, with the caller-owned object set
        //     widened from the call's to the batch's
        if let Err(e) = self.wait_fence(fence) {
            let idle = self.wait_idle();
            if e == vk::Result::ERROR_DEVICE_LOST || idle == Err(vk::Result::ERROR_DEVICE_LOST) {
                self.poison(
                    format!("device lost after a failed fence wait on the deferred batch ({what}) (wait {e:?}, idle {idle:?}); nothing runs on it again"),
                    true,
                );
                self.release_retired(retired, retired_kernels);
                self.release_staging(retired_staging);
                self.release_submission(fence, free_cb);
                return Err(AckError::Vulkan(vk::Result::ERROR_DEVICE_LOST));
            }
            return match idle {
                Ok(()) => {
                    self.release_retired(retired, retired_kernels);
                    self.release_staging(retired_staging);
                    self.release_submission(fence, free_cb);
                    Err(e.into())
                }
                Err(other) => {
                    let why = format!(
                        "the flush of the deferred batch ({what}) failed its fence wait ({e:?}) and the device could not be quiesced ({other:?}): EVERY operation in the batch is of unknown completion, none is known to have run and none is known not to have run; the batch's fence, command buffer and retired objects stay allocated, and this context refuses every later call"
                    );
                    self.poison(why.clone(), false);
                    // the fence (1), the command buffer (1) and every object
                    // retired into this batch stay allocated, counted through
                    // the one place that decides destroy-or-leak
                    self.leaked.set(self.leaked.get() + 2);
                    self.release_retired(retired, retired_kernels);
                    self.release_staging(retired_staging);
                    Err(AckError::Poisoned(why))
                }
            };
        }
        // the fence has signalled, so every stamp this batch wrote is in the
        // pool; read them before the queries are reset by the next batch
        let host_elapsed_ns = timing_started.elapsed().as_nanos() as f64;
        if stamps > 1 && !ts_overflow && self.accumulate_timestamps(stamps, host_elapsed_ns) {
            self.ts_unresolved.set(self.ts_unresolved.get() - 1);
        }
        self.release_retired(retired, retired_kernels);
        self.release_staging(retired_staging);
        self.release_submission(fence, free_cb);
        Ok(())
    }

    /// Read this batch's `stamps` timestamps back and fold them into the
    /// instrument's totals.
    ///
    /// Stamp 0 precedes the first dispatch; each later stamp marks completion
    /// of the preceding commands. Deltas are ordered completion intervals,
    /// not exact per-item service times. Their sum is the batch's device span,
    /// excluding transfers, queue waits and host work outside that batch.
    /// The remainder of wall time is not a CPU-only measurement.
    ///
    /// WHAT IT IS NOT. Not a utilisation or occupancy figure, and not a
    /// separation of arithmetic from memory traffic; both need a different
    /// instrument. It is summed per item rather than taken as
    /// `last - first` with the whole submission's host bound guarding
    /// counter-wrap ambiguity.
    /// An ambiguous batch contributes no interval and remains unresolved.
    ///
    /// A FAILED READBACK IS COUNTED, NOT SWALLOWED: `ts_unresolved` moves and
    /// nothing is added, so a run whose device time is partly unreadable
    /// cannot report a short span as a complete one.
    fn accumulate_timestamps(&self, stamps: u32, host_elapsed_ns: f64) -> bool {
        let qp = self.ts_pool.get();
        let mut ticks = vec![0u64; stamps as usize];
        let read = unsafe {
            self.device.get_query_pool_results(
                qp,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if read.is_err() {
            return false;
        }
        let Ok((sum_ns, min_ns, max_ns)) = summarize_dispatch_timestamps(
            &ticks,
            self.timestamp_valid_bits,
            f64::from(self.report.timestamp_period_ns),
            host_elapsed_ns,
        ) else {
            return false;
        };
        self.ts_flushes.set(self.ts_flushes.get() + 1);
        self.ts_items
            .set(self.ts_items.get() + (ticks.len() as u64 - 1));
        self.ts_span_ns
            .set(self.ts_span_ns.get().saturating_add(sum_ns));
        self.ts_min_item_ns
            .set(self.ts_min_item_ns.get().min(min_ns));
        self.ts_max_item_ns
            .set(self.ts_max_item_ns.get().max(max_ns));
        true
    }

    /// Switch the timestamp instrument on or off for THIS context, whatever
    /// `ACK_DISPATCH_TIMESTAMPS` said at open.
    ///
    /// REFUSED WITH A BATCH OPEN, by name. Switching mid-batch would leave the
    /// batch's items partly stamped, and a span read off a stamped PREFIX is a
    /// wrong number rather than a missing one. The caller flushes first (a
    /// readback, or `sync`) and then switches.
    pub fn set_dispatch_timestamps(&self, on: bool) -> Result<()> {
        if self.recorder.borrow().is_some() {
            return Err(AckError::Unsupported(format!(
                "the dispatch timestamp instrument cannot be switched to {on} with {} item(s) recorded and unsubmitted: the batch would be stamped in part and its span would cover a prefix",
                self.recorded_items()
            )));
        }
        self.ts_enabled.set(on);
        Ok(())
    }

    /// What the dispatch instrument has counted and timed; see
    /// `DispatchStats` for what each figure is and is not.
    pub fn dispatch_stats(&self) -> DispatchStats {
        let timed_items = self.ts_items.get();
        DispatchStats {
            items: self.disp_items.get(),
            commands: self.disp_commands.get(),
            barriers: self.disp_barriers.get(),
            flushes: self.flushes(),
            timed_flushes: self.ts_flushes.get(),
            timed_items,
            device_ns: self.ts_span_ns.get(),
            // a minimum of u64::MAX means nothing was timed; report 0 rather
            // than a sentinel a caller might arithmetic on
            min_item_ns: if timed_items == 0 {
                0
            } else {
                self.ts_min_item_ns.get()
            },
            max_item_ns: self.ts_max_item_ns.get(),
            unresolved_flushes: self.ts_unresolved.get(),
            timestamps_enabled: self.ts_enabled.get(),
        }
    }

    /// A flush that failed at or before `vkQueueSubmit`: the batch is gone.
    ///
    /// WHAT THIS DOES NOT CLAIM. `vkQueueSubmit` guarantees that a failure
    /// left the queue and every resource unaffected only for
    /// `VK_ERROR_OUT_OF_HOST_MEMORY` and `VK_ERROR_OUT_OF_DEVICE_MEMORY`; it
    /// is required to return `VK_ERROR_DEVICE_LOST` precisely when it cannot
    /// make that guarantee. So "nothing was submitted, so none of its
    /// operations ran" is derivable for the out-of-memory results and for the
    /// two calls that precede the submission, and is NOT derivable for a lost
    /// device. The message therefore says which of the two it is instead of
    /// asserting a provenance this code cannot establish, and a lost device
    /// takes the established-loss arm with the real `VkResult`.
    ///
    /// WHY IT POISONS EVEN WHEN THE DEVICE IS HEALTHY. Every dispatch in this
    /// batch was reported ACCEPTED to its caller and none of them ran; the
    /// command buffer is freed here and a batch is never retried, so they
    /// cannot be recovered. The buffers they were to write still hold their
    /// previous bytes, and `ffi`'s "a readback always sees every dispatch
    /// accepted before it" would otherwise be silently false for the rest of
    /// this context's life. Refusing every later call by name is the only
    /// outcome that is not a wrong answer with an alibi. The objects are
    /// released BEFORE the poison is set: nothing was submitted, so nothing
    /// can be using them, and `release_or_leak`'s conservative leak rule is
    /// for objects whose device state is unknown, which these are not.
    fn lose_batch(&self, rec: Recorder, e: vk::Result, what: &str, step: &str) -> AckError {
        self.release_retired(rec.retired, rec.retired_kernels);
        self.release_staging(rec.retired_staging);
        self.vk_destroy_fence(rec.fence);
        self.vk_free_command_buffer(rec.cb);
        if e == vk::Result::ERROR_DEVICE_LOST {
            self.poison(
                format!(
                    "device lost at {step} of the deferred batch ({what}); whether any of its \
                     operations ran is unknown, and nothing runs on this device again"
                ),
                true,
            );
            return AckError::Vulkan(vk::Result::ERROR_DEVICE_LOST);
        }
        let ran = match e {
            vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                "nothing was submitted, so none of its operations ran"
            }
            // outside the results Vulkan scopes its unaffected-resources
            // guarantee to: do not assert what was or was not submitted
            _ => {
                "whether any of its operations ran is UNKNOWN: this result is outside the set \
                  for which the submission is guaranteed to have left the queue unaffected"
            }
        };
        let why = format!(
            "flush ({what}) failed at {step}: {e:?}; {ran}. Every operation in this batch was \
             already reported accepted and none of them can now run, so this context is poisoned \
             and refuses every later call"
        );
        self.poison(why.clone(), false);
        AckError::Batch(why)
    }

    /// Drop the recorded batch without submitting it.
    ///
    /// Commands that were never submitted never run, so this is safe. It is
    /// public so a teardown path can release the batch BEFORE it destroys the
    /// kernels whose descriptor sets those recorded commands name, rather than
    /// relying on the order two `Drop` implementations happen to run in. It is
    /// not a flush and it is not a substitute for one: work a caller was told
    /// was accepted is discarded, so `ffi::ack_close` flushes instead.
    pub fn discard_recorded_batch(&self) {
        self.abandon_batch();
    }

    /// Abandon the open batch without submitting it: recorded commands that
    /// were never submitted never run, so dropping them is safe.
    fn abandon_batch(&self) {
        // the batch in flight names kernels and buffers the caller may be about
        // to destroy: seen through first (a poisoned context leaks it, counted)
        let in_flight = self.in_flight.borrow_mut().take();
        if let Some(sub) = in_flight {
            if self.is_poisoned() {
                self.leaked.set(
                    self.leaked.get()
                        + 2
                        + 2 * sub.retired.len() as u64
                        + 4 * sub.retired_kernels.len() as u64
                        + 2 * sub.retired_staging.len() as u64,
                );
            } else {
                let _ = self.finish(sub);
            }
        }
        let taken = self.recorder.borrow_mut().take();
        if let Some(rec) = taken {
            for r in rec.retired {
                // the commands that named them are being dropped unsubmitted,
                // so nothing can run against these pairs and the pool may hold
                // them
                self.recycle_or_destroy(r.buffer, r.memory, r.allocated);
            }
            // the commands that named them are being dropped unsubmitted, so
            // the kernels a caller retired into this batch can go too
            for k in rec.retired_kernels {
                self.vk_destroy_descriptor_pool(k.pool);
                self.vk_destroy_pipeline(k.pipeline);
                self.vk_destroy_pipeline_layout(k.layout);
                self.vk_destroy_descriptor_set_layout(k.dsl);
            }
            // the copies that read them were never submitted
            for (b, m) in rec.retired_staging {
                self.vk_destroy_buffer(b);
                self.vk_free_memory(m);
            }
            self.vk_destroy_fence(rec.fence);
            self.vk_free_command_buffer(rec.cb);
        }
    }

    /// Arm a fault for the next wait, idle or creation of the named kind. The
    /// real Vulkan call still runs first: for a wait or an idle the GPU work
    /// has completed when the substituted result is acted on; for a creation
    /// the wrapper destroys the object the real call made (or, for the bind
    /// and the descriptor set, leaves the real effect and substitutes only the
    /// result), so the caller's cleanup path runs with nothing of the
    /// injection's own left behind.
    #[cfg(feature = "fault-injection")]
    pub fn inject_fault(&self, fault: Fault) {
        self.fault.borrow_mut().push(fault);
    }

    /// Faults armed and not yet consumed; a refusal that precedes every
    /// creation leaves a creation fault armed (tests read it).
    #[cfg(feature = "fault-injection")]
    pub fn armed_faults(&self) -> usize {
        self.fault.borrow().len()
    }

    #[cfg(feature = "fault-injection")]
    pub fn clear_faults(&self) {
        self.fault.borrow_mut().clear();
    }

    fn wait_fence(&self, fence: vk::Fence) -> std::result::Result<(), vk::Result> {
        let real = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) };
        #[cfg(feature = "fault-injection")]
        {
            let mut faults = self.fault.borrow_mut();
            if let Some(i) = faults.iter().position(|f| matches!(f, Fault::WaitFails(_))) {
                if let Fault::WaitFails(r) = faults.remove(i) {
                    return Err(r);
                }
            }
        }
        real
    }

    fn wait_idle(&self) -> std::result::Result<(), vk::Result> {
        let real = unsafe { self.device.device_wait_idle() };
        #[cfg(feature = "fault-injection")]
        {
            let mut faults = self.fault.borrow_mut();
            if let Some(i) = faults.iter().position(|f| matches!(f, Fault::IdleFails(_))) {
                if let Fault::IdleFails(r) = faults.remove(i) {
                    return Err(r);
                }
            }
        }
        real
    }

    /// The identity resources created here carry.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Vulkan objects this context's methods created and have not destroyed
    /// (ACK-L0-05); counted by the create and destroy wrappers, never stamped
    /// beside them. It includes caller-owned objects, retired objects, cached
    /// buffer pairs and the optional context-owned timestamp query pool.
    /// `leaked_on_poison()` counts only objects encountered by the explicit
    /// leak-accounting paths, so it cannot reconstruct this total. On a
    /// healthy or lost context `leaked_on_poison()` is 0.
    /// The objects `open` creates before a context
    /// exists (instance, device, command pool) are outside the count.
    pub fn live_objects(&self) -> i64 {
        self.live_objects.get()
    }

    #[cfg(feature = "fault-injection")]
    fn take_create_fault(&self, step: CreateStep) -> Option<vk::Result> {
        let mut faults = self.fault.borrow_mut();
        let i = faults
            .iter()
            .position(|f| matches!(f, Fault::CreateFails(s, _) if *s == step))?;
        match faults.remove(i) {
            Fault::CreateFails(_, r) => Some(r),
            _ => None,
        }
    }

    #[cfg(not(feature = "fault-injection"))]
    fn take_create_fault(&self, _step: CreateStep) -> Option<vk::Result> {
        None
    }

    fn track(&self, delta: i64) {
        self.live_objects.set(self.live_objects.get() + delta);
    }

    // ---- the create and destroy wrappers: one call, one count, one place a
    // fault can be injected after the real creation succeeded (ACK-L0-05)

    fn vk_create_buffer(
        &self,
        info: &vk::BufferCreateInfo,
    ) -> std::result::Result<vk::Buffer, vk::Result> {
        let b = unsafe { self.device.create_buffer(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::Buffer) {
            self.vk_destroy_buffer(b);
            return Err(r);
        }
        Ok(b)
    }

    fn vk_destroy_buffer(&self, b: vk::Buffer) {
        unsafe { self.device.destroy_buffer(b, None) };
        self.track(-1);
    }

    fn vk_allocate_memory(
        &self,
        info: &vk::MemoryAllocateInfo,
    ) -> std::result::Result<vk::DeviceMemory, vk::Result> {
        let m = unsafe { self.device.allocate_memory(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::Memory) {
            self.vk_free_memory(m);
            return Err(r);
        }
        Ok(m)
    }

    fn vk_free_memory(&self, m: vk::DeviceMemory) {
        unsafe { self.device.free_memory(m, None) };
        self.track(-1);
    }

    fn vk_bind_buffer_memory(
        &self,
        b: vk::Buffer,
        m: vk::DeviceMemory,
    ) -> std::result::Result<(), vk::Result> {
        unsafe { self.device.bind_buffer_memory(b, m, 0) }?;
        match self.take_create_fault(CreateStep::BindMemory) {
            Some(r) => Err(r),
            None => Ok(()),
        }
    }

    fn vk_create_shader_module(
        &self,
        info: &vk::ShaderModuleCreateInfo,
    ) -> std::result::Result<vk::ShaderModule, vk::Result> {
        let m = unsafe { self.device.create_shader_module(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::ShaderModule) {
            self.vk_destroy_shader_module(m);
            return Err(r);
        }
        Ok(m)
    }

    fn vk_destroy_shader_module(&self, m: vk::ShaderModule) {
        unsafe { self.device.destroy_shader_module(m, None) };
        self.track(-1);
    }

    fn vk_create_descriptor_set_layout(
        &self,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> std::result::Result<vk::DescriptorSetLayout, vk::Result> {
        let l = unsafe { self.device.create_descriptor_set_layout(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::DescriptorSetLayout) {
            self.vk_destroy_descriptor_set_layout(l);
            return Err(r);
        }
        Ok(l)
    }

    fn vk_destroy_descriptor_set_layout(&self, l: vk::DescriptorSetLayout) {
        unsafe { self.device.destroy_descriptor_set_layout(l, None) };
        self.track(-1);
    }

    fn vk_create_pipeline_layout(
        &self,
        info: &vk::PipelineLayoutCreateInfo,
    ) -> std::result::Result<vk::PipelineLayout, vk::Result> {
        let l = unsafe { self.device.create_pipeline_layout(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::PipelineLayout) {
            self.vk_destroy_pipeline_layout(l);
            return Err(r);
        }
        Ok(l)
    }

    fn vk_destroy_pipeline_layout(&self, l: vk::PipelineLayout) {
        unsafe { self.device.destroy_pipeline_layout(l, None) };
        self.track(-1);
    }

    fn vk_create_compute_pipeline(
        &self,
        info: &[vk::ComputePipelineCreateInfo],
    ) -> std::result::Result<vk::Pipeline, vk::Result> {
        let p = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), info, None)
        }
        .map_err(|(_, r)| r)?[0];
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::Pipeline) {
            self.vk_destroy_pipeline(p);
            return Err(r);
        }
        Ok(p)
    }

    fn vk_destroy_pipeline(&self, p: vk::Pipeline) {
        unsafe { self.device.destroy_pipeline(p, None) };
        self.track(-1);
    }

    fn vk_create_descriptor_pool(
        &self,
        info: &vk::DescriptorPoolCreateInfo,
    ) -> std::result::Result<vk::DescriptorPool, vk::Result> {
        let p = unsafe { self.device.create_descriptor_pool(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::DescriptorPool) {
            self.vk_destroy_descriptor_pool(p);
            return Err(r);
        }
        Ok(p)
    }

    fn vk_destroy_descriptor_pool(&self, p: vk::DescriptorPool) {
        unsafe { self.device.destroy_descriptor_pool(p, None) };
        self.track(-1);
    }

    /// A descriptor set lives in its pool and is not counted; an injected
    /// failure after the real allocation is covered by the pool's destruction.
    ///
    /// One call allocates the kernel's whole arena (`sets_per_kernel_pool`
    /// sets), so the object count per kernel is unchanged at four -- the pool,
    /// the pipeline, the pipeline layout and the set layout -- however many
    /// sets the arena holds.
    fn vk_allocate_descriptor_sets(
        &self,
        info: &vk::DescriptorSetAllocateInfo,
    ) -> std::result::Result<Vec<vk::DescriptorSet>, vk::Result> {
        let s = unsafe { self.device.allocate_descriptor_sets(info) }?;
        match self.take_create_fault(CreateStep::DescriptorSet) {
            Some(r) => Err(r),
            None => Ok(s),
        }
    }

    fn vk_create_fence(&self) -> std::result::Result<vk::Fence, vk::Result> {
        let f = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::Fence) {
            self.vk_destroy_fence(f);
            return Err(r);
        }
        Ok(f)
    }

    fn vk_destroy_fence(&self, f: vk::Fence) {
        unsafe { self.device.destroy_fence(f, None) };
        self.track(-1);
    }

    fn vk_allocate_command_buffer(&self) -> std::result::Result<vk::CommandBuffer, vk::Result> {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { self.device.allocate_command_buffers(&alloc) }?[0];
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::CommandBuffer) {
            self.vk_free_command_buffer(cb);
            return Err(r);
        }
        Ok(cb)
    }

    fn vk_free_command_buffer(&self, cb: vk::CommandBuffer) {
        unsafe { self.device.free_command_buffers(self.command_pool, &[cb]) };
        self.track(-1);
    }

    fn vk_create_query_pool(
        &self,
        info: &vk::QueryPoolCreateInfo,
    ) -> std::result::Result<vk::QueryPool, vk::Result> {
        let q = unsafe { self.device.create_query_pool(info, None) }?;
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::QueryPool) {
            self.vk_destroy_query_pool(q);
            return Err(r);
        }
        Ok(q)
    }

    fn vk_destroy_query_pool(&self, q: vk::QueryPool) {
        unsafe { self.device.destroy_query_pool(q, None) };
        self.track(-1);
    }

    fn check_buffer(&self, b: &Buffer) -> Result<()> {
        if b.ctx_id != self.id {
            return Err(AckError::Foreign(format!(
                "buffer {} belongs to context {}, not context {}",
                b.id, b.ctx_id, self.id
            )));
        }
        if !self.live_buffers.borrow().contains(&b.id) {
            return Err(AckError::Freed(format!(
                "buffer {} of context {} was destroyed",
                b.id, self.id
            )));
        }
        Ok(())
    }

    fn check_kernel(&self, k: &Kernel) -> Result<()> {
        if k.ctx_id != self.id {
            return Err(AckError::Foreign(format!(
                "kernel belongs to context {}, not context {}",
                k.ctx_id, self.id
            )));
        }
        Ok(())
    }

    fn memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.memory.memory_type_count {
            let supported = requirements.memory_type_bits & (1 << i) != 0;
            let ok = self.memory.memory_types[i as usize]
                .property_flags
                .contains(flags);
            if supported && ok {
                return Ok(i);
            }
        }
        Err(AckError::Unsupported(format!(
            "no memory type with {flags:?}"
        )))
    }

    fn raw_buffer(
        &self,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.vk_create_buffer(&info)?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        // ACK-L0-05: a failure after the buffer exists destroys it; a failure
        // after the memory exists frees that too
        let type_index = match self.memory_type(req, flags) {
            Ok(i) => i,
            Err(e) => {
                self.vk_destroy_buffer(buffer);
                return Err(e);
            }
        };
        if let Some(r) = self.take_create_fault(CreateStep::MemoryType) {
            self.vk_destroy_buffer(buffer);
            return Err(AckError::Vulkan(r));
        }
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(type_index);
        let memory = match self.vk_allocate_memory(&alloc) {
            Ok(m) => m,
            Err(e) => {
                self.vk_destroy_buffer(buffer);
                return Err(e.into());
            }
        };
        if let Err(e) = self.vk_bind_buffer_memory(buffer, memory) {
            self.vk_free_memory(memory);
            self.vk_destroy_buffer(buffer);
            return Err(e.into());
        }
        Ok((buffer, memory))
    }

    /// A device-local storage buffer. Zero-byte and over-limit requests are
    /// refused before any Vulkan call (review finding ACK-L0-01).
    pub fn buffer(&self, bytes: u64) -> Result<Buffer> {
        self.check_live()?;
        // The VkBuffer is created a whole 32-bit word long even when the
        // logical size is not, so a shader reading this buffer through a `uint`
        // view can always read the word holding the last element. The bf16
        // edge matmul does exactly that, and the word holding a final element
        // at an even index is half past an unrounded size. `bytes` below stays
        // the LOGICAL size, so upload and download keep checking exactly what
        // the caller asked for; the padding is never read for its value, only
        // as the discarded half of a word.
        let allocated = bytes.next_multiple_of(4);
        if bytes == 0 || allocated > self.max_storage_buffer_bytes {
            return Err(AckError::Unsupported(format!(
                "buffer of {bytes} bytes is outside (0, {}]",
                self.max_storage_buffer_bytes
            )));
        }
        // THE REFUSALS ABOVE STAY ABOVE THE POOL LOOKUP. `tests/leaks.rs`
        // asserts that a zero-size request leaves an armed creation fault
        // ARMED, which is a statement about this ordering; a pool consulted
        // first would answer a refusable request.
        //
        // A HIT MINTS A FRESH `id`, ALWAYS. `check_dispatch` decides a bound
        // buffer is still live by `live_buffers.contains(&id)` and `Kernel`
        // stores ids, not handles, so recycling an id would revalidate a stale
        // binding: a kernel bound to a dead buffer would pass the liveness
        // check and dispatch against the new owner's tensor, reporting ACK_OK.
        // `next_buffer` is monotonic for the life of the context and this does
        // not change that.
        let (buffer, memory) = match self.take_pooled(allocated) {
            Some(p) => {
                self.pool_hits.set(self.pool_hits.get() + 1);
                (p.buffer, p.memory)
            }
            None => {
                self.pool_misses.set(self.pool_misses.get() + 1);
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST;
                let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                match self.raw_buffer(allocated, usage, flags) {
                    Ok(pair) => pair,
                    Err(e) => {
                        // THE POOL MUST NEVER BE THE REASON AN ALLOCATION
                        // FAILED. It is holding device memory that nothing
                        // owns, so an allocation failure with a non-empty pool
                        // is the one case where the right answer is certain:
                        // give the memory back and try once more. Without this
                        // the ceiling is a SAFETY mechanism and has to be set
                        // low; with it the ceiling is a pure tuning knob, and
                        // that is what lets the default be raised. The retry
                        // is once, not a loop: a second failure after the pool
                        // is empty is a real out-of-memory and is returned.
                        if self.pooled_objects() == 0 {
                            return Err(e);
                        }
                        self.clear_pool();
                        self.pool_reclaims.set(self.pool_reclaims.get() + 1);
                        self.raw_buffer(allocated, usage, flags)?
                    }
                }
            }
        };
        let id = self.next_buffer.get();
        self.next_buffer.set(id + 1);
        self.live_buffers.borrow_mut().insert(id);
        Ok(Buffer {
            buffer,
            memory,
            bytes,
            ctx_id: self.id,
            id,
        })
    }

    /// Record, submit and wait for one command buffer. Before the submission
    /// succeeds every error path frees what it created. After it, a failed
    /// wait quiesces the device with `device_wait_idle`: idle success releases
    /// the fence and command buffer and returns the wait error; an established
    /// device loss (reported by the wait or by the idle) releases them and
    /// poisons the context; any other idle error leaves them allocated, poisons
    /// the context and returns `Poisoned` (review finding ACK-L0-03).
    ///
    /// This is no longer every submission the crate makes. It is `upload`'s,
    /// `download`'s and `dispatch_timed`'s; a recorded batch is submitted by
    /// `flush`, which repeats this ladder with "the call's objects" widened to
    /// "the batch's objects".
    fn one_shot<F: FnOnce(vk::CommandBuffer)>(&self, record: F) -> Result<()> {
        self.check_live()?;
        let cb = self.vk_allocate_command_buffer()?;
        let cbs = [cb];
        let free_cb = || self.vk_free_command_buffer(cb);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { self.device.begin_command_buffer(cb, &begin) } {
            free_cb();
            return Err(e.into());
        }
        record(cb);
        if let Err(e) = unsafe { self.device.end_command_buffer(cb) } {
            free_cb();
            return Err(e.into());
        }
        let fence = match self.vk_create_fence() {
            Ok(f) => f,
            Err(e) => {
                free_cb();
                return Err(e.into());
            }
        };
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, &submit, fence) } {
            self.vk_destroy_fence(fence);
            free_cb();
            return Err(e.into());
        }
        if let Err(e) = self.wait_fence(fence) {
            // the submission may be in flight: only a quiesced device or an
            // established device loss makes releasing its fence and command
            // buffer safe (ACK-L0-03). Anything else leaves them allocated and
            // poisons the context, because its state is unknown from here on.
            // A wait that itself reports the device lost is an established
            // loss whatever the idle says afterwards: a lost device stays lost.
            let idle = self.wait_idle();
            if e == vk::Result::ERROR_DEVICE_LOST || idle == Err(vk::Result::ERROR_DEVICE_LOST) {
                self.poison(
                    format!("device lost after a failed fence wait (wait {e:?}, idle {idle:?}); nothing runs on it again"),
                    true,
                );
                self.release_submission(fence, free_cb);
                return Err(AckError::Vulkan(vk::Result::ERROR_DEVICE_LOST));
            }
            return match idle {
                Ok(()) => {
                    self.release_submission(fence, free_cb);
                    Err(e.into())
                }
                Err(other) => {
                    let why = format!(
                        "fence wait failed ({e:?}) and the device could not be quiesced ({other:?}): the submission may still be running, its fence and command buffer stay allocated, and this context refuses every later call"
                    );
                    self.poison(why.clone(), false);
                    // the fence and the command buffer stay allocated
                    self.leaked.set(self.leaked.get() + 2);
                    Err(AckError::Poisoned(why))
                }
            };
        }
        self.release_submission(fence, free_cb);
        Ok(())
    }

    /// Copy host bytes into a device-local buffer through a staging buffer. The
    /// slice must be exactly the buffer's size: a short or long copy is refused
    /// rather than partially applied (review finding ACK-L0-01).
    ///
    /// EAGER, AND A FLUSH POINT. Phase 1 of deferred submission records
    /// dispatches only; a transfer is submitted here and waited for, exactly
    /// as before. It flushes the recorded batch first
    /// (`FlushReason::Upload`), because its own submission would otherwise run
    /// ahead of dispatches recorded before it. Its refusals, its staging
    /// lifetime and its ACK-L0-03 ladder are unchanged, which is why the
    /// poison and leak suites still drive them through this method.
    pub fn upload(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        self.check_live()?;
        self.check_buffer(dst)?;
        let bytes = data.len() as u64;
        if bytes == 0 || bytes != dst.bytes {
            return Err(AckError::Unsupported(format!(
                "upload of {bytes} bytes into a {}-byte buffer",
                dst.bytes
            )));
        }
        if self.deferred_upload() {
            return self.upload_recorded(dst, data);
        }
        // after every named refusal, before any object of this call exists
        self.flush(batch::FlushReason::Upload)?;
        let (staging, memory) = self.raw_buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let release = || {
            self.vk_destroy_buffer(staging);
            self.vk_free_memory(memory);
        };
        unsafe {
            let p = match self
                .device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
            {
                Ok(p) => p,
                Err(e) => {
                    // nothing was submitted: the staging pair is safe to destroy
                    release();
                    return Err(e.into());
                }
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u8>(), data.len());
            self.device.unmap_memory(memory);
        }
        let submitted = self.one_shot(|cb| unsafe {
            let region = [vk::BufferCopy::default().size(bytes)];
            self.device
                .cmd_copy_buffer(cb, staging, dst.buffer, &region);
        });
        // the staging pair is this method's: released when the device is
        // quiesced or lost, left allocated and counted when the context is
        // poisoned with the device state unknown
        self.release_or_leak(2, release);
        submitted
    }

    /// The recorded upload (2026-09-10): the bytes go into a staging buffer
    /// now, the copy into `dst` is recorded into the open batch after a
    /// barrier whenever the batch already holds a command, and the staging
    /// pair retires into the batch, destroyed when its fence signals and never
    /// pooled. Nothing is flushed and nothing is waited for: an upload no
    /// longer drains the queue, and a later flush or readback carries it. The
    /// caller's bytes are copied before this returns, as before. The copy is
    /// not an item of the plan: it counts toward neither the depth nor the
    /// instrument's items and barriers, and the next dispatch takes a barrier
    /// after it whether or not it is the batch's first.
    fn upload_recorded(&self, dst: &Buffer, data: &[u8]) -> Result<()> {
        let bytes = data.len() as u64;
        let (staging, memory) = self.raw_buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let release = || {
            self.vk_destroy_buffer(staging);
            self.vk_free_memory(memory);
        };
        unsafe {
            let p = match self
                .device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
            {
                Ok(p) => p,
                Err(e) => {
                    // nothing was recorded: the staging pair is safe to destroy
                    release();
                    return Err(e.into());
                }
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u8>(), data.len());
            self.device.unmap_memory(memory);
        }
        if let Err(e) = self.open_batch() {
            release();
            return Err(e);
        }
        let due = {
            let mut slot = self.recorder.borrow_mut();
            let rec = slot
                .as_mut()
                .expect("open_batch installed the recorder or returned an error");
            let cb = rec.cb;
            // the copy WRITES `dst`: a barrier before it when an item since the
            // last barrier touched that buffer, or when another transfer is
            // already pending (two copies into one batch are ordered by one)
            if (rec.plan.record_transfer_write(dst.id) || rec.pending_transfer)
                && self.barriers_enabled()
            {
                self.record_barrier(cb);
            }
            unsafe {
                let region = [vk::BufferCopy::default().size(bytes)];
                self.device
                    .cmd_copy_buffer(cb, staging, dst.buffer, &region);
            }
            rec.pending_transfer = true;
            rec.retired_staging.push((staging, memory));
            rec.plan.due()
        };
        if let Some(reason) = due {
            self.flush(reason)?;
        }
        Ok(())
    }

    /// Host-visible memory property flags for READBACK staging, best first.
    ///
    /// `HOST_CACHED` is the one that matters: without it the mapping is
    /// write-combined on this driver and a CPU read from it bypasses the cache,
    /// which measured as a flat 0.20 GiB/s regardless of transfer size. The
    /// first entry adds `HOST_COHERENT` so no invalidate is needed; the second
    /// is cached but not coherent, which `download` handles by invalidating
    /// before the read; the third is the pre-2026-09-08 behaviour, kept last so
    /// a device exposing no cached host-visible type still works — slowly and
    /// correctly, rather than not at all.
    pub(crate) fn readback_memory_preferences() -> [vk::MemoryPropertyFlags; 3] {
        [
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]
    }

    /// Copy a device-local buffer back to the host.
    ///
    /// THE STAGING MEMORY HERE IS CHOSEN FOR CPU **READS**, which is why it does
    /// not use `upload`'s flags. `HOST_VISIBLE | HOST_COHERENT` without
    /// `HOST_CACHED` is write-combined on this driver: ideal for the writes
    /// `upload` does, and pathological for the reads this method does, because
    /// a CPU read from write-combined memory bypasses the cache entirely — no
    /// prefetch, no line reuse.
    ///
    /// Measured on the RX 9070 XT before this change, with pre-allocated
    /// destinations and a device-to-device control:
    ///
    /// | size | upload | download | device to device |
    /// |---|---|---|---|
    /// | 1 MiB | 1.69 GiB/s | 0.38 GiB/s | 9.88 GiB/s |
    /// | 4 MiB | 4.82 | 0.20 | 39.15 |
    /// | 16 MiB | 7.25 | 0.20 | 107.48 |
    /// | 64 MiB | 8.27 | 0.20 | 153.88 |
    ///
    /// Download was **flat** at 0.20 GiB/s across a 64x size range. Flat is the
    /// diagnosis: a bandwidth ceiling still amortises fixed cost and improves
    /// with size, so a rate that does not move with size is per-byte work at a
    /// fixed rate, not a transfer limit.
    ///
    /// THE FLUSH POINT THAT MAKES DEFERRAL SAFE RATHER THAN AN OPTIMISATION.
    /// The copy must see every dispatch recorded before it, and the host must
    /// see the copy, so nothing may be reordered across a readback
    /// (`FlushReason::Readback`). Every host fallback in the torch adapter
    /// reaches this method, so the gain from deferral is bounded by the
    /// fallback count of a step: increment 5 and a zero-fallback step multiply
    /// rather than add. Whether that bound bites on this card is UNMEASURED.
    pub fn download(&self, src: &Buffer) -> Result<Vec<u8>> {
        self.check_live()?;
        self.check_buffer(src)?;
        // after every named refusal, before any object of this call exists
        self.flush(batch::FlushReason::Readback)?;
        let bytes = src.bytes;
        // Preference order, best first. `HOST_CACHED` is what makes the read
        // fast; `HOST_COHERENT` alongside it saves an invalidate. The last entry
        // is the pre-2026-09-08 behaviour, kept as a fallback so a device
        // exposing no cached host-visible type still works — slowly, and
        // correctly, rather than not at all.
        let preferences = Self::readback_memory_preferences();
        let mut chosen: Option<(vk::Buffer, vk::DeviceMemory, vk::MemoryPropertyFlags)> = None;
        let mut last_err = None;
        for flags in preferences {
            match self.raw_buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST, flags) {
                Ok((b, m)) => {
                    chosen = Some((b, m, flags));
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let (staging, memory, staging_flags) = match chosen {
            Some(t) => t,
            None => {
                return Err(last_err.unwrap_or_else(|| {
                    AckError::Unsupported("no host-visible memory type for readback".into())
                }))
            }
        };
        let coherent = staging_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        let release = || {
            self.vk_destroy_buffer(staging);
            self.vk_free_memory(memory);
        };
        let submitted = self.one_shot(|cb| unsafe {
            let region = [vk::BufferCopy::default().size(bytes)];
            self.device
                .cmd_copy_buffer(cb, src.buffer, staging, &region);
        });
        if let Err(e) = submitted {
            // as in upload: destroy when quiesced or lost, leak and count when poisoned
            self.release_or_leak(2, release);
            return Err(e);
        }
        let mut out = vec![0u8; bytes as usize];
        unsafe {
            // the submission completed: the staging pair is safe to destroy on every path below
            let p = match self
                .device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
            {
                Ok(p) => p,
                Err(e) => {
                    release();
                    return Err(e.into());
                }
            };
            // A non-coherent cached mapping must be invalidated before the read,
            // or the CPU may serve it from stale cache lines. WHOLE_SIZE is used
            // rather than `bytes` because a non-whole range must be aligned to
            // nonCoherentAtomSize and WHOLE_SIZE carries no such requirement.
            if !coherent {
                let range = [vk::MappedMemoryRange::default()
                    .memory(memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                if let Err(e) = self.device.invalidate_mapped_memory_ranges(&range) {
                    self.device.unmap_memory(memory);
                    release();
                    return Err(e.into());
                }
            }
            std::ptr::copy_nonoverlapping(p.cast::<u8>(), out.as_mut_ptr(), out.len());
            self.device.unmap_memory(memory);
        }
        release();
        Ok(out)
    }

    /// Build a compute pipeline from SPIR-V with `storage_buffers` storage-buffer
    /// bindings (set 0, bindings 0..n) and `push_bytes` of push constants.
    pub fn kernel(&self, spirv: &[u8], storage_buffers: u32, push_bytes: u32) -> Result<Kernel> {
        self.kernel_with_subgroup(spirv, storage_buffers, push_bytes, None)
    }

    /// As `kernel`, but the pipeline is created with a required subgroup size
    /// and full subgroups, for kernels written for one subgroup width.
    pub fn kernel_with_subgroup(
        &self,
        spirv: &[u8],
        storage_buffers: u32,
        push_bytes: u32,
        required_subgroup_size: Option<u32>,
    ) -> Result<Kernel> {
        self.check_live()?;
        // ACK-L0-01 residual: the push range is validated before any object exists
        if push_bytes % 4 != 0 || push_bytes.max(4) > self.max_push_constant_bytes {
            return Err(AckError::Unsupported(format!(
                "push constant range of {push_bytes} bytes is not a multiple of 4 within the device's {}",
                self.max_push_constant_bytes
            )));
        }
        if required_subgroup_size.is_some() && !self.subgroup_size_control {
            // ACK-L0-05: refused before any object exists
            return Err(AckError::Unsupported(
                "kernel requires a fixed subgroup size but the device has no subgroupSizeControl"
                    .into(),
            ));
        }
        let words = ash::util::read_spv(&mut Cursor::new(spirv))
            .map_err(|e| AckError::Io(format!("{e}")))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // ACK-L0-05: everything created below is owned by the build until the
        // kernel takes it; a failure at any step drops the build, which
        // destroys what exists
        let mut build = KernelBuild::new(self);
        build.module = Some(self.vk_create_shader_module(&module_info)?);
        let module = build.module.unwrap();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..storage_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = self.vk_create_descriptor_set_layout(&dsl_info)?;
        build.dsl = Some(dsl);
        let dsls = [dsl];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_bytes.max(4))];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&dsls)
            .push_constant_ranges(&push);
        let layout = self.vk_create_pipeline_layout(&layout_info)?;
        build.layout = Some(layout);
        let entry = c"main";
        let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(required_subgroup_size.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        if required_subgroup_size.is_some() {
            stage = stage
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .push_next(&mut required);
        }
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = self.vk_create_compute_pipeline(&pipeline_info)?;
        build.pipeline = Some(pipeline);
        // THE DESCRIPTOR ARENA. Before deferred submission a kernel had ONE
        // descriptor set, and `bind` rewrote it. That was safe for exactly one
        // reason: every dispatch was submitted and waited for on its own, so
        // the set was free again before the next bind. Recording two
        // dispatches of one kernel into one batch falsifies that: rewriting
        // the set for the second while the first is recorded and unsubmitted
        // makes the first execute against the second's buffers -- a wrong
        // answer with no error, and a Vulkan validity violation. So the pool
        // holds `sets_per_kernel_pool` sets, allocated once here, handed out
        // one per recorded dispatch and reusable only after a flush.
        // two banks of `arena` sets: a batch records into one bank while the
        // batch in flight executes against the other (asynchronous flushes)
        let arena = self.sets_per_kernel_pool.max(1);
        let banks = 2 * arena;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(banks * storage_buffers.max(1))];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(banks)
            .pool_sizes(&pool_sizes);
        let pool = self.vk_create_descriptor_pool(&pool_info)?;
        build.pool = Some(pool);
        let arena_layouts = vec![dsl; banks as usize];
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&arena_layouts);
        let sets = self.vk_allocate_descriptor_sets(&set_info)?;
        // the module served its purpose; the kernel takes the rest from the build
        self.vk_destroy_shader_module(build.module.take().unwrap());
        build.take_all();
        Ok(Kernel {
            pipeline,
            layout,
            dsl,
            pool,
            sets,
            next_set: Cell::new(0),
            set_gen: Cell::new(0),
            storage_buffers,
            push_bytes: push_bytes.max(4),
            ctx_id: self.id,
            bound: RefCell::new(Vec::new()),
            pending: RefCell::new(Vec::new()),
        })
    }

    /// Bind buffers to the kernel's storage bindings in order. Every buffer
    /// must be this context's and live; the kernel remembers what it holds so
    /// a dispatch after a destroy is refused (ACK-L0-02).
    ///
    /// THIS MAKES NO VULKAN CALL. It validates -- the same four checks in the
    /// same order with the same names as before -- and records the binding on
    /// the kernel. The descriptor set is written at the dispatch, against the
    /// set that dispatch takes from the kernel's arena, because a set written
    /// here would be rewritten by the next bind while a recorded, unsubmitted
    /// dispatch still referred to it. `bind` therefore has no error path but
    /// its validations, and loses none of them.
    pub fn bind(&self, kernel: &Kernel, buffers: &[&Buffer]) -> Result<()> {
        self.check_live()?;
        self.check_kernel(kernel)?;
        for b in buffers {
            self.check_buffer(b)?;
        }
        if buffers.len() as u32 != kernel.storage_buffers {
            return Err(AckError::Unsupported(format!(
                "kernel expects {} buffers, got {}",
                kernel.storage_buffers,
                buffers.len()
            )));
        }
        *kernel.pending.borrow_mut() = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.bytes)
            })
            .collect();
        *kernel.bound.borrow_mut() = buffers.iter().map(|b| b.id).collect();
        Ok(())
    }

    /// Take the next descriptor set of a kernel's arena, or `None` when the
    /// arena is exhausted by recorded, unsubmitted dispatches. A kernel whose
    /// watermark belongs to an earlier batch generation starts again at set 0:
    /// the batch that held its sets has been flushed.
    fn acquire_set(&self, kernel: &Kernel) -> Option<vk::DescriptorSet> {
        if !self.arena_enabled() {
            // the sabotage: the one set this kernel had before deferral
            return Some(kernel.sets[0]);
        }
        if kernel.set_gen.get() != self.batch_gen.get() {
            kernel.next_set.set(0);
            kernel.set_gen.set(self.batch_gen.get());
        }
        // the arena is one bank; the pool holds two, and the batch's bank is
        // the one the batch in flight is not executing against
        let bank_len = kernel.sets.len() / 2;
        let i = batch::next_set_index(kernel.next_set.get(), bank_len as u32)?;
        kernel.next_set.set(i + 1);
        Some(kernel.sets[self.open_bank() * bank_len + i as usize])
    }

    /// Sets of `kernel` handed out since the last flush.
    pub fn descriptor_watermark(&self, kernel: &Kernel) -> u32 {
        if kernel.set_gen.get() != self.batch_gen.get() {
            return 0;
        }
        kernel.next_set.get()
    }

    /// Write the binding `bind` recorded into one descriptor set.
    fn write_set(&self, kernel: &Kernel, set: vk::DescriptorSet) {
        let infos = kernel.pending.borrow();
        if infos.is_empty() {
            return;
        }
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Every check a dispatch makes before it touches Vulkan, in one place so
    /// `dispatch` and `dispatch_timed` cannot drift apart in what they refuse,
    /// in what order, or under what name.
    fn check_dispatch(
        &self,
        kernel: &Kernel,
        push: &[u8],
        groups: [u32; 3],
        repeats: u32,
        what: &str,
    ) -> Result<()> {
        self.check_live()?;
        self.check_kernel(kernel)?;
        // ACK-L0-02: a kernel dispatches only with the buffers it was bound to,
        // all of them still live; an unbound set is a Vulkan validity violation
        {
            let bound = kernel.bound.borrow();
            if kernel.storage_buffers > 0 && bound.is_empty() {
                return Err(AckError::Unsupported(
                    "dispatch of a kernel with no buffers bound".into(),
                ));
            }
            let live = self.live_buffers.borrow();
            if let Some(id) = bound.iter().find(|id| !live.contains(id)) {
                return Err(AckError::Freed(format!(
                    "buffer {id} bound to the kernel was destroyed before dispatch"
                )));
            }
        }
        // ACK-L0-01 residual: the push constants must fit the range the layout
        // declared; an empty slice is a kernel with no push constants and is
        // simply not pushed
        if push.len() % 4 != 0 || push.len() as u32 > kernel.push_bytes {
            return Err(AckError::Unsupported(format!(
                "push constants of {} bytes do not fit the kernel's {}-byte range (multiple of 4)",
                push.len(),
                kernel.push_bytes
            )));
        }
        // ACK-L0-01: a batch of zero dispatches would time nothing and report it
        // as a measurement; an empty or oversized grid is a Vulkan validity violation
        if repeats == 0 {
            return Err(AckError::Unsupported(format!(
                "{what} needs at least one repeat"
            )));
        }
        for (axis, &count) in groups.iter().enumerate() {
            if count == 0 || count > self.max_workgroup_count[axis] {
                return Err(AckError::Unsupported(format!(
                    "workgroup count {count} on axis {axis} is outside [1, {}]",
                    self.max_workgroup_count[axis]
                )));
            }
        }
        Ok(())
    }

    /// Record `repeats` back-to-back launches of `kernel` into the deferred
    /// batch, with a compute-to-compute barrier between launches.
    ///
    /// RETURNS WHEN THE DISPATCH IS RECORDED, NOT WHEN IT HAS RUN. The work is
    /// submitted at the next flush (`batch::FlushReason` is the whole list of
    /// reasons), and a failure of that flush is reported to whoever triggers
    /// it, naming the batch rather than an operation (see `flush`). Every
    /// check `dispatch_timed` makes before touching Vulkan is made here first,
    /// in the same order, under the same names.
    ///
    /// `access` declares, slot for slot, which buffers this dispatch reads and
    /// writes. It is checked against what the kernel is bound to and is
    /// otherwise inert: R1 records a barrier between every pair of adjacent
    /// items whatever they touch, so nothing reads the declaration (see
    /// `batch::check_declared_access`).
    pub fn dispatch(
        &self,
        kernel: &Kernel,
        push: &[u8],
        groups: [u32; 3],
        repeats: u32,
        access: &[Access],
    ) -> Result<()> {
        self.check_dispatch(kernel, push, groups, repeats, "dispatch")?;
        {
            let bound = kernel.bound.borrow();
            if let Err(why) = batch::check_declared_access(access, &bound) {
                return Err(AckError::Unsupported(why));
            }
        }
        // the arena, with one flush and exactly one retry; a second failure is
        // an error, never a loop
        let set = match self.acquire_set(kernel) {
            Some(s) => s,
            None => {
                self.flush(batch::FlushReason::DescriptorsExhausted)?;
                match self.acquire_set(kernel) {
                    Some(s) => s,
                    None => {
                        return Err(AckError::Unsupported(format!(
                            "the kernel's descriptor arena of {} set(s) is exhausted after a flush",
                            kernel.sets.len() / 2
                        )))
                    }
                }
            }
        };
        self.open_batch()?;
        self.write_set(kernel, set);
        {
            let mut slot = self.recorder.borrow_mut();
            let rec = slot
                .as_mut()
                .expect("open_batch installed the recorder or returned an error");
            // The barrier rule (2026-09-10): before an item whose DECLARED
            // accesses conflict with the items since the last barrier, and
            // after a recorded upload whatever the item's place in the batch.
            // `check_declared_access` above has already refused a declaration
            // that does not match what the kernel is bound to, which is what
            // makes it safe to decide on it. ACK_BARRIERS=always restores the
            // unconditional rule.
            let r1 = rec.plan.record_with(access);
            let needs_barrier = r1 || rec.pending_transfer;
            rec.pending_transfer = false;
            let cb = rec.cb;
            if needs_barrier && self.barriers_enabled() {
                self.record_barrier(cb);
            }
            unsafe {
                self.device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    &[set],
                    &[],
                );
                if !push.is_empty() {
                    self.device.cmd_push_constants(
                        cb,
                        kernel.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push,
                    );
                }
                let repeat_barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                for i in 0..repeats {
                    if i > 0 {
                        self.device.cmd_pipeline_barrier(
                            cb,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &repeat_barrier,
                            &[],
                            &[],
                        );
                    }
                    self.device
                        .cmd_dispatch(cb, groups[0], groups[1], groups[2]);
                }
            }
            // THE INSTRUMENT. Three increments counting what reached the
            // command buffer, and -- only when it is switched on -- one
            // timestamp marking this item's completion. The stamp is
            // BOTTOM_OF_PIPE and not TOP_OF_PIPE: a top-of-pipe stamp latches
            // when earlier commands REACH that stage, which a compute barrier
            // does not block, so it is not a start-of-dispatch marker. Under
            // R1 -- one full memory barrier between every adjacent pair --
            // consecutive bottom-of-pipe stamps are ordered completion
            // markers, not exact per-item service times: later compute can
            // overlap the preceding marker. The origin dependency above
            // establishes the start of the aggregate batch span.
            self.disp_items.set(self.disp_items.get() + 1);
            self.disp_commands
                .set(self.disp_commands.get() + u64::from(repeats));
            // the instrument counts R1's barriers (items - 1 per batch) and the
            // repeat barriers; the barrier after a recorded upload is the
            // transfer's, not an item's, and is not counted
            self.disp_barriers.set(
                self.disp_barriers.get()
                    + u64::from(r1 && self.barriers_enabled())
                    + u64::from(repeats.saturating_sub(1)),
            );
            if self.ts_enabled.get() && rec.stamps > 0 {
                if rec.stamps < self.ts_capacity.get() {
                    unsafe {
                        self.device.cmd_write_timestamp(
                            cb,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            self.ts_pool.get(),
                            rec.stamps,
                        )
                    };
                    rec.stamps += 1;
                } else {
                    rec.ts_overflow = true;
                }
            }
        }
        self.next_item_seq.set(self.next_item_seq.get() + 1);
        // THE BUDGET IS CHECKED AFTER THE ITEM IS RECORDED, so a budget of one
        // item flushes at the end of every call and the deferred path is the
        // eager arm -- one code path, no mode branch that can drift
        let due = self.recorder.borrow().as_ref().and_then(|r| r.plan.due());
        if let Some(reason) = due {
            self.flush(reason)?;
        }
        Ok(())
    }

    /// Dispatch `repeats` back-to-back launches of `kernel` with the given push
    /// constants and workgroup grid, with a compute-to-compute barrier between
    /// launches, and return the GPU time of the whole batch in milliseconds
    /// from timestamp queries (not wall clock).
    ///
    /// UNCHANGED IN MEANING under deferred submission: this submits its own
    /// dispatch, alone and fenced, and the duration it returns is that
    /// dispatch's. It FLUSHES the recorded batch first
    /// (`FlushReason::TimedDispatch`) so the measurement covers this dispatch
    /// and nothing recorded before it, and the host clock the timestamp
    /// resolution is bounded by starts after that flush, so it still bounds
    /// only this submission.
    pub fn dispatch_timed(
        &self,
        kernel: &Kernel,
        push: &[u8],
        groups: [u32; 3],
        repeats: u32,
    ) -> Result<f64> {
        self.check_dispatch(kernel, push, groups, repeats, "dispatch_timed")?;
        // everything recorded so far is submitted and waited for, so the
        // duration below is this dispatch's and not the batch's
        self.flush(batch::FlushReason::TimedDispatch)?;
        let set = match self.acquire_set(kernel) {
            Some(s) => s,
            None => {
                self.flush(batch::FlushReason::DescriptorsExhausted)?;
                match self.acquire_set(kernel) {
                    Some(s) => s,
                    None => {
                        return Err(AckError::Unsupported(format!(
                            "the kernel's descriptor arena of {} set(s) is exhausted after a flush",
                            kernel.sets.len() / 2
                        )))
                    }
                }
            }
        };
        self.write_set(kernel, set);
        let qp_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        let qp = self.vk_create_query_pool(&qp_info)?;
        // ACK-L0-04: the host's elapsed time around the submission bounds the
        // device duration from above and decides whether a wrapped counter can
        // be resolved at all (see `resolve_timestamps`)
        let started = std::time::Instant::now();
        let submitted = self.one_shot(|cb| unsafe {
            self.device.cmd_reset_query_pool(cb, qp, 0, 2);
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                kernel.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    cb,
                    kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            self.device
                .cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            for i in 0..repeats {
                if i > 0 {
                    self.device.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                }
                self.device
                    .cmd_dispatch(cb, groups[0], groups[1], groups[2]);
            }
            self.device
                .cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
        });
        if let Err(e) = submitted {
            // the query pool is this method's: destroyed when quiesced or lost, leaked and counted when poisoned
            self.release_or_leak(1, || self.vk_destroy_query_pool(qp));
            return Err(e);
        }
        let mut ticks = [0u64; 2];
        let read = unsafe {
            self.device.get_query_pool_results(
                qp,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        // the submission completed: the pool is safe to destroy whatever the read said
        self.vk_destroy_query_pool(qp);
        read?;
        let host_ns = started.elapsed().as_secs_f64() * 1.0e9;
        let ns = resolve_timestamps(
            ticks[0],
            ticks[1],
            self.timestamp_valid_bits,
            f64::from(self.report.timestamp_period_ns),
            host_ns,
        )?;
        Ok(ns / 1.0e6)
    }

    /// Destroy a buffer this context created. Destroying another context's
    /// buffer is a contract violation in safe code and panics before any
    /// Vulkan call rather than freeing memory on the wrong device.
    ///
    /// WITH A BATCH OPEN THE OBJECTS ARE RETIRED, NOT DESTROYED. Recorded,
    /// unsubmitted commands may still name this `VkBuffer`, so it is put on
    /// the batch's retirement list and destroyed by the flush. The id leaves
    /// `live_buffers` immediately either way, so a dispatch on it is still
    /// refused `Freed` at once; `live_objects()` counts the retired pair until
    /// the flush destroys it. Making the free itself a flush point was
    /// rejected: the torch adapter frees a kit buffer on every tensor death,
    /// so it would flush hundreds of times a step and remove the whole gain.
    /// The retirement budget (`FlushReason::RetiredBytes`) is what keeps the
    /// held memory bounded instead.
    ///
    /// THIS DELETER CANNOT RAISE, AND WHAT IT DROPS IS NAMED. Crossing the
    /// retirement budget makes it flush, and a flush can fail. Of the ways it
    /// can, every one that LOSES work poisons the context, so the next call
    /// refuses by name and nothing is silently swallowed; the one that does
    /// not (a fence wait that failed on a device that then quiesced -- the
    /// batch ran, only the wait's diagnostic is in doubt) has its error
    /// dropped here and returned by `release_buffer`, which is the entry the C
    /// ABI uses. A caller that wants the flush's result calls that one.
    pub fn destroy_buffer(&self, b: Buffer) {
        let _ = self.free_buffer(b);
    }

    /// `destroy_buffer`'s body, with the retirement flush's result kept.
    fn free_buffer(&self, b: Buffer) -> Result<()> {
        assert!(
            b.ctx_id == self.id,
            "destroy_buffer: buffer {} belongs to context {}, not context {}",
            b.id,
            b.ctx_id,
            self.id
        );
        self.live_buffers.borrow_mut().remove(&b.id);
        let allocated = b.bytes.next_multiple_of(4);
        // `PoolMode::Immediate` is a SABOTAGE and this is where it applies: it
        // takes the pair back while a batch that may name it is still open and
        // unsubmitted. Nothing else in the function changes.
        if !self.pool_defers_to_retirement() {
            self.release_or_leak(2, || {
                self.recycle_or_destroy(b.buffer, b.memory, allocated);
            });
            return Ok(());
        }
        let due = {
            let mut slot = self.recorder.borrow_mut();
            match slot.as_mut() {
                Some(rec) => {
                    rec.retired.push(RetiredBuffer {
                        buffer: b.buffer,
                        memory: b.memory,
                        allocated,
                    });
                    rec.plan.retire(b.bytes, 2)
                }
                None => {
                    // nothing is recorded, but the batch in flight may name it:
                    // then the pair is destroyed when that batch's fence signals
                    let mut in_flight = self.in_flight.borrow_mut();
                    if let Some(sub) = in_flight.as_mut() {
                        sub.retired.push(RetiredBuffer {
                            buffer: b.buffer,
                            memory: b.memory,
                            allocated,
                        });
                    } else {
                        // nothing is recorded and nothing is in flight: the pool
                        // may take it back at once. Work MAY still be running on a
                        // poisoned context, in which case leaking the pair -- not
                        // pooling it -- is the safe outcome, which is why the
                        // recycle is inside `release_or_leak` and not beside it.
                        self.release_or_leak(2, || {
                            self.recycle_or_destroy(b.buffer, b.memory, allocated);
                        });
                    }
                    None
                }
            }
        };
        if let Some(reason) = due {
            self.flush(reason)?;
        }
        Ok(())
    }

    /// `destroy_buffer` that reports whether the buffer was destroyed or, on a
    /// poisoned context with the device state unknown, left allocated. The C
    /// ABI turns the latter into `ACK_ERR_POISONED`, so a caller's accounting
    /// (the torch adapter's `free_refused`) sees the poison instead of a free
    /// that reported success for an object it leaked.
    ///
    /// A RETIRED BUFFER REPORTS `Ok`: the handle is consumed and the objects
    /// are destroyed at the next flush. If that flush later poisons the
    /// context with the device state unknown, the retired objects are leaked
    /// and counted then -- so a free that reported success can still end up as
    /// a leak, attributed to the flush rather than to this call. That is the
    /// same widening of attribution the batch makes everywhere else (see
    /// `flush`), stated here rather than hidden.
    ///
    /// A FAILURE OF THE RETIREMENT FLUSH IS RETURNED, NOT SWALLOWED. Freeing a
    /// buffer past `max_retired_bytes` flushes the batch, and this entry
    /// reports that flush's error rather than dropping it -- the handle is
    /// consumed either way, and the alternative is a device error nothing ever
    /// reports. That widens the CIRCUMSTANCES of the codes `ack_buffer_free`
    /// already returns (`-2`, `-10`); it adds no code and moves no signature.
    /// The leak report takes precedence when both apply, because "the objects
    /// were leaked" is the more specific statement about this call.
    pub fn release_buffer(&self, b: Buffer) -> Result<()> {
        let before = self.leaked.get();
        let flushed = self.free_buffer(b);
        if self.leaked.get() != before {
            let why = self.poisoned.borrow().clone().unwrap_or_default();
            return Err(AckError::Poisoned(why));
        }
        flushed
    }

    /// Destroy a kernel this context created. Destroying another context's
    /// kernel is a contract violation in safe code and panics before any
    /// Vulkan call.
    ///
    /// WITH A BATCH OPEN THE OBJECTS ARE RETIRED, NOT DESTROYED, exactly as
    /// `destroy_buffer`'s are and for the same reason. A recorded, unsubmitted
    /// dispatch holds `cmd_bind_pipeline` on this pipeline and
    /// `cmd_bind_descriptor_sets` on a set from this pool; destroying either
    /// moves that command buffer to the invalid state, and the flush's
    /// `vkQueueSubmit` on an invalid command buffer is undefined behaviour,
    /// not an error this crate could report. So the four objects go on the
    /// batch's retirement list and the flush destroys them. `live_objects()`
    /// counts them until then, and the kernel handle is consumed at once
    /// either way, so no later call can name it.
    ///
    /// Unlike a buffer, a kernel is not device memory, so retiring one adds
    /// nothing to `max_retired_bytes` and can never itself trigger a flush;
    /// four objects are added to the batch's retired-object count.
    ///
    /// Before deferred submission this method's body was safe with no such
    /// list, because every dispatch was submitted and waited for inside
    /// `dispatch_timed` before it could return. `ffi::Inner::drop` guards the
    /// teardown path a second way -- it discards the recorded batch before its
    /// destroy loop -- and that guard is kept; this makes the safe Rust API
    /// safe on its own rather than by the order a caller happens to use.
    pub fn destroy_kernel(&self, k: Kernel) {
        assert!(
            k.ctx_id == self.id,
            "destroy_kernel: kernel belongs to context {}, not context {}",
            k.ctx_id,
            self.id
        );
        let retired = {
            let mut slot = self.recorder.borrow_mut();
            match slot.as_mut() {
                Some(rec) => {
                    rec.retired_kernels.push(RetiredKernel {
                        pipeline: k.pipeline,
                        layout: k.layout,
                        dsl: k.dsl,
                        pool: k.pool,
                    });
                    // no bytes: a kernel is not device memory, so it cannot
                    // cross the retirement budget on its own
                    rec.plan.retire(0, 4);
                    true
                }
                None => match self.in_flight.borrow_mut().as_mut() {
                    // the batch in flight may name it: destroyed at its fence
                    Some(sub) => {
                        sub.retired_kernels.push(RetiredKernel {
                            pipeline: k.pipeline,
                            layout: k.layout,
                            dsl: k.dsl,
                            pool: k.pool,
                        });
                        true
                    }
                    None => false,
                },
            }
        };
        if !retired {
            // nothing is recorded, so nothing can name it; work may still be
            // in flight from a poisoned context, in which case leaking the
            // four objects is the safe outcome
            self.release_or_leak(4, || {
                self.vk_destroy_descriptor_pool(k.pool);
                self.vk_destroy_pipeline(k.pipeline);
                self.vk_destroy_pipeline_layout(k.layout);
                self.vk_destroy_descriptor_set_layout(k.dsl);
            });
        }
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub fn physical(&self) -> vk::PhysicalDevice {
        self.physical
    }
}

/// Two device timestamps into a duration in nanoseconds (ACK-L0-04). The
/// counter is `valid_bits` wide and wraps there, not at 64: one wrap is
/// resolved by reducing the difference modulo 2^bits, but two or more cannot
/// be told from none, so the host's own elapsed time around the submission,
/// which bounds the device duration from above (the host waited for the
/// fence), decides: a batch whose host time reaches the counter's period is
/// reported as `AmbiguousTiming` instead of modulo the period; the dispatch
/// itself completed, so callers that only need the product treat that error
/// as success with no duration (the C ABI does). With 64 valid bits the
/// period is longer than any run and the check never fires.
pub fn resolve_timestamps(
    t0: u64,
    t1: u64,
    valid_bits: u32,
    period_ns: f64,
    host_elapsed_ns: f64,
) -> Result<f64> {
    let delta = t1.wrapping_sub(t0);
    if valid_bits >= 64 {
        return Ok(delta as f64 * period_ns);
    }
    let period_ticks = 1u64 << valid_bits;
    let wrap_ns = period_ticks as f64 * period_ns;
    if host_elapsed_ns >= wrap_ns {
        return Err(AckError::AmbiguousTiming(format!(
            "timed dispatch took {host_elapsed_ns:.0} ns by the host clock, at or beyond the {valid_bits}-bit timestamp counter's period of {wrap_ns:.0} ns: the device duration cannot be resolved"
        )));
    }
    Ok((delta & (period_ticks - 1)) as f64 * period_ns)
}

/// Completion intervals are accepted together. The host bound covers the
/// whole submission; a possible extra counter wrap makes its entire span
/// unresolved, even when every modulo interval happens to look small.
fn summarize_dispatch_timestamps(
    ticks: &[u64],
    valid_bits: u32,
    period_ns: f64,
    host_elapsed_ns: f64,
) -> Result<(u64, u64, u64)> {
    let mut sum = 0u64;
    let mut min = u64::MAX;
    let mut max = 0;
    for interval in ticks.windows(2) {
        let ns = resolve_timestamps(
            interval[0],
            interval[1],
            valid_bits,
            period_ns,
            host_elapsed_ns,
        )? as u64;
        sum = sum.saturating_add(ns);
        min = min.min(ns);
        max = max.max(ns);
    }
    Ok((sum, min, max))
}

/// The objects a kernel build has created so far; dropping it destroys them
/// (ACK-L0-05). `take_all` hands them to the `Kernel` on success.
struct KernelBuild<'a> {
    ctx: &'a Context,
    module: Option<vk::ShaderModule>,
    dsl: Option<vk::DescriptorSetLayout>,
    layout: Option<vk::PipelineLayout>,
    pipeline: Option<vk::Pipeline>,
    pool: Option<vk::DescriptorPool>,
}

impl<'a> KernelBuild<'a> {
    fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            module: None,
            dsl: None,
            layout: None,
            pipeline: None,
            pool: None,
        }
    }

    fn take_all(&mut self) {
        self.module = None;
        self.dsl = None;
        self.layout = None;
        self.pipeline = None;
        self.pool = None;
    }
}

impl Drop for KernelBuild<'_> {
    fn drop(&mut self) {
        // reverse order of creation; the descriptor set, if any, dies with its pool
        if let Some(p) = self.pool.take() {
            self.ctx.vk_destroy_descriptor_pool(p);
        }
        if let Some(p) = self.pipeline.take() {
            self.ctx.vk_destroy_pipeline(p);
        }
        if let Some(l) = self.layout.take() {
            self.ctx.vk_destroy_pipeline_layout(l);
        }
        if let Some(d) = self.dsl.take() {
            self.ctx.vk_destroy_descriptor_set_layout(d);
        }
        if let Some(m) = self.module.take() {
            self.ctx.vk_destroy_shader_module(m);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // a poisoned context whose device is not known to be lost may still be
        // executing: its pool, device, instance and loader handle are leaked
        // rather than destroyed under running work (ACK-L0-03); a lost device
        // drops all work, so its teardown is permitted
        if self.is_poisoned() && !self.lost.get() {
            return;
        }
        // A recorded batch is NOT flushed here: commands that were never
        // submitted never run, so abandoning them is safe, and submitting work
        // a caller did not ask for at teardown would be worse. The idle comes
        // first (it covers work that WAS submitted), then the abandon frees
        // the batch's fence, command buffer and retired buffers, so nothing of
        // it outlives the pool.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.abandon_batch();
        // the pool holds live Vulkan objects; drain them here, after the
        // abandon that may have added to it and before the device goes, or the
        // validation layer reports them at `vkDestroyDevice` under the
        // signature this crate assigns to a caller contract violation
        self.clear_pool();
        // the instrument's query pool, if it was ever created: after the idle
        // above, so no submitted batch can still be writing into it
        let qp = self.ts_pool.get();
        if qp != vk::QueryPool::null() {
            self.vk_destroy_query_pool(qp);
            self.ts_pool.set(vk::QueryPool::null());
        }
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            std::mem::ManuallyDrop::drop(&mut self._entry);
        }
    }
}

/// What the dispatch instrument has counted, read through
/// `Context::dispatch_stats` and through the C entry `ack_dispatch_stats`.
///
/// WHY IT EXISTS. The decomposition of the registered training step
/// (`docs/audits/2026-09-08-dispatch-instrumentation.md`) put 51.2% of a
/// microbatch pass in the transformer layers' non-product elementwise and
/// reduce work and 33% in a per-dispatch fixed cost, and then could not choose
/// between fusing those chains and letting independent dispatches overlap,
/// because nothing in this crate reported how many dispatches a pass issues or
/// how long the device spent on them. These are those two numbers.
///
/// WHAT `device_ns` IS: device-elapsed time, summed over items, between an
/// ordered batch origin and completion markers. It includes
/// barrier drain, wave launch, kernel arithmetic and memory traffic, but the
/// intervals are not exact per-item service times. Wall time outside this
/// span also includes device transfers and queue waits. WHAT IT IS NOT:
/// a utilisation figure, an occupancy figure, or a split of arithmetic from
/// memory traffic.
///
/// The counts always run. `device_ns` and the per-item extremes are zero
/// unless `ACK_DISPATCH_TIMESTAMPS` was set when the device was opened, which
/// `timestamps_enabled` reports, so an absent figure is distinguishable from a
/// measured zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchStats {
    /// Items recorded by `Context::dispatch`.
    pub items: u64,
    /// `vkCmdDispatch` commands recorded; `>= items`, because one item with
    /// `repeats > 1` records one command per repeat.
    pub commands: u64,
    /// Pipeline barriers recorded: inter-item/repeat barriers plus the
    /// instrument's origin dependency when enabled.
    pub barriers: u64,
    /// Every flush of this context, whatever the reason (`Context::flushes`).
    pub flushes: u64,
    /// Flushes whose timestamps were read back and folded in.
    pub timed_flushes: u64,
    /// Items covered by those readbacks: the denominator for `device_ns`.
    pub timed_items: u64,
    /// Device-elapsed nanoseconds over `timed_items`.
    pub device_ns: u64,
    /// The shortest and the longest single item's device-elapsed time; 0 when
    /// nothing was timed.
    pub min_item_ns: u64,
    pub max_item_ns: u64,
    /// Flushes that carried stamps whose span was NOT folded in: the
    /// submission/wait/readback failed, a wrap was ambiguous, or the batch
    /// outran the query pool. Nonzero means `device_ns`
    /// covers less than the run.
    pub unresolved_flushes: u64,
    /// Whether the timestamp instrument was switched on at open.
    pub timestamps_enabled: bool,
}

/// What the buffer pool has done, read through `Context::pool_stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// `Context::buffer` calls served without a Vulkan allocation.
    pub hits: u64,
    /// `Context::buffer` calls that allocated.
    pub misses: u64,
    /// Freed pairs the pool took back.
    pub returns: u64,
    /// Freed pairs destroyed because holding them would cross the ceiling.
    pub declined: u64,
    /// Pairs destroyed by a drain (teardown, disable, or a lowered ceiling).
    pub drained: u64,
    /// Allocation failures that drained the pool and retried, so a pool
    /// holding unowned memory can never be the reason an allocation failed.
    pub reclaims: u64,
    /// Vulkan objects the pool holds (two per pair).
    pub held_objects: i64,
    /// Allocated bytes the pool holds.
    pub held_bytes: u64,
}

/// A device-local storage buffer; it knows which context made it.
pub struct Buffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub bytes: u64,
    ctx_id: u64,
    id: u64,
}

impl Buffer {
    /// The creating context's id.
    pub fn context_id(&self) -> u64 {
        self.ctx_id
    }

    /// This buffer's id within its context (the C ABI's sequence number is
    /// separate; this one is the safe runtime's).
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// A compute pipeline with an arena of descriptor sets of storage buffers; it
/// knows which context made it and which buffers it was last bound to.
///
/// The arena is what makes deferred submission safe for two dispatches of one
/// kernel in one batch: each recorded dispatch takes its own set, and a set is
/// reusable only after the flush that ran the dispatch which held it. The
/// kernel is still four counted Vulkan objects (pool, pipeline, pipeline
/// layout, set layout); descriptor sets live in their pool and are not counted.
pub struct Kernel {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// `Context::sets_per_kernel_pool` sets, allocated once at build time.
    sets: Vec<vk::DescriptorSet>,
    /// The next set to hand out, within the current batch generation.
    next_set: Cell<u32>,
    /// The context batch generation `next_set` belongs to. A generation older
    /// than the context's means every set of this kernel is free again,
    /// because the batch that held them has been flushed.
    set_gen: Cell<u64>,
    storage_buffers: u32,
    push_bytes: u32,
    ctx_id: u64,
    bound: RefCell<Vec<u64>>,
    /// What `bind` recorded: the descriptor writes a dispatch applies to the
    /// set it takes. `bind` makes no Vulkan call, so the write happens at the
    /// dispatch, against that dispatch's own set.
    pending: RefCell<Vec<vk::DescriptorBufferInfo>>,
}

impl Kernel {
    pub fn context_id(&self) -> u64 {
        self.ctx_id
    }

    /// Descriptor sets one batch of this kernel can hand out (the arena); the
    /// pool holds two such banks, one per batch in flight or being recorded.
    pub fn descriptor_sets(&self) -> u32 {
        (self.sets.len() / 2) as u32
    }
}

/// Round-to-nearest-even f32 -> bf16 bits; preserve special classes and quiet NaNs.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let is_nan = (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0;
    if is_nan {
        // Rounding a NaN payload can carry into the sign or erase its mantissa.
        // Keep the sign/high payload and set the BF16 quiet bit instead.
        return ((b >> 16) | 0x0040) as u16;
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
mod tests {
    #[test]
    fn dispatch_timing_resolves_one_wrap_but_refuses_an_ambiguous_batch() {
        // An 8-bit clock at 2 ns/tick: these completion stamps cross zero.
        assert_eq!(
            super::summarize_dispatch_timestamps(&[250, 3, 10], 8, 2.0, 100.0).unwrap(),
            (32, 14, 18)
        );
        // Identical stamps cannot reveal an additional period of execution.
        assert!(matches!(
            super::summarize_dispatch_timestamps(&[250, 3, 10], 8, 2.0, 512.0),
            Err(super::AckError::AmbiguousTiming(_))
        ));
    }

    #[test]
    fn readback_staging_prefers_cached_host_memory() {
        // The defect this pins: `download` allocated its readback staging with
        // upload's flags, HOST_VISIBLE | HOST_COHERENT. That is write-combined
        // on this driver, and a CPU read from write-combined memory bypasses
        // the cache -- measured on the RX 9070 XT as a FLAT 0.20 GiB/s across a
        // 64x size range, against 4-8 GiB/s for upload through the same code
        // path and 154 GiB/s device to device. Adding HOST_CACHED took the
        // 64 MiB readback to 3.04 GiB/s and made it scale with size again.
        //
        // A throughput assertion here would be a claim about the scheduler, so
        // this asserts the STRUCTURE instead: every preference that is tried
        // before the legacy fallback must ask for HOST_CACHED, and the legacy
        // entry must be last. Reordering these, or dropping HOST_CACHED, is the
        // regression, and it is invisible in any functional test because the
        // slow path returns exactly the same bytes.
        use super::vk::MemoryPropertyFlags as F;
        let prefs = super::Context::readback_memory_preferences();
        assert!(
            prefs[..prefs.len() - 1]
                .iter()
                .all(|f| f.contains(F::HOST_CACHED)),
            "every preference before the fallback must request HOST_CACHED: {prefs:?}"
        );
        assert!(
            prefs.iter().all(|f| f.contains(F::HOST_VISIBLE)),
            "readback staging must be host visible: {prefs:?}"
        );
        assert!(
            !prefs[prefs.len() - 1].contains(F::HOST_CACHED),
            "the last entry is the uncached fallback, kept so a device with no              cached host-visible type still works"
        );
        assert!(
            prefs[0].contains(F::HOST_COHERENT),
            "the first choice is coherent so the read needs no invalidate"
        );
    }

    use super::{bf16_to_f32, f32_to_bf16, Context};

    #[test]
    fn a_context_is_send_so_it_can_live_behind_a_mutex() {
        fn assert_send<T: Send>() {}
        assert_send::<Context>();
        // the not-Sync half of the contract is the compile_fail doctest on Context
    }

    #[test]
    fn bf16_round_trip_is_exact_for_bf16_representable_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 1.5, 3.0, 65280.0, -0.125] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v);
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1 + 2^-8 sits exactly between 1.0 and 1 + 2^-7 in bf16: ties go to even (1.0)
        let tie = 1.0f32 + 2.0f32.powi(-8);
        assert_eq!(bf16_to_f32(f32_to_bf16(tie)), 1.0);
        // 1 + 3*2^-9 is above the tie point: rounds up to 1 + 2^-7
        let above = 1.0f32 + 3.0 * 2.0f32.powi(-9);
        assert_eq!(bf16_to_f32(f32_to_bf16(above)), 1.0 + 2.0f32.powi(-7));
    }

    #[test]
    fn bf16_conversion_preserves_signed_zeros_infinities_and_finite_ties() {
        let cases = [
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x7f80_0000, 0x7f80),
            (0xff80_0000, 0xff80),
            (0x3f80_8000, 0x3f80),
            (0x3f81_8000, 0x3f82),
            (0xbf80_8000, 0xbf80),
            (0xbf81_8000, 0xbf82),
            (0x0000_8000, 0x0000),
            (0x0000_8001, 0x0001),
            (0x8000_8000, 0x8000),
            (0x8000_8001, 0x8001),
        ];
        for (input, expected) in cases {
            assert_eq!(f32_to_bf16(f32::from_bits(input)), expected);
        }
    }

    #[test]
    fn bf16_conversion_preserves_nan_sign_and_high_payload_while_quieting() {
        for sign in [0, 0x8000_0000] {
            for high_payload in 0..128 {
                for low_payload in [0, 1, 0x7fff, 0x8000, 0xffff] {
                    if high_payload == 0 && low_payload == 0 {
                        continue;
                    }
                    let input_bits = sign | 0x7f80_0000 | (high_payload << 16) | low_payload;
                    let converted = f32_to_bf16(f32::from_bits(input_bits));
                    assert_eq!(converted & 0x7f80, 0x7f80, "NaN exponent changed");
                    assert_ne!(converted & 0x007f, 0, "NaN became infinity");
                    assert_ne!(converted & 0x0040, 0, "NaN was not quieted");
                    assert_eq!(
                        converted,
                        ((input_bits >> 16) | 0x0040) as u16,
                        "NaN sign or high payload changed"
                    );
                }
            }
        }
    }
}
