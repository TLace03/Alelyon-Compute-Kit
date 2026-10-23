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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::io::Cursor;
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub mod amp_ops;
pub mod attention_fused;
pub mod attention_ops;
pub mod batch;
pub mod convert_ops;
pub mod device_choice;
pub mod embedding_ops;
pub mod expert_factor_ops;
pub mod ffi;
pub mod head_loss_ops;
pub mod index_ops;
pub mod intwise_ops;
pub mod loss_ops;
pub mod matmul_ops;
pub mod nf4_ops;
pub mod pointwise_ops;
pub mod radix_ops;
pub mod reduce_ops;
pub mod rotary_ops;
pub mod row_ops;
pub mod student_norm_ops;
pub mod swiglu_ops;
pub mod vq4_ops;
pub mod weighted_bag_ops;

pub use batch::{Access, AccessKind, FlushReason, ALL_FLUSH_REASONS, FLUSH_REASON_COUNT};

// ---- the transfer clock (off by default) ------------------------------------
//
// A process-wide switch and cumulative counters. On, `Context::download_into`
// and `Context::upload` time their flush (the submission of whatever is
// recorded, and the wait for it and for any batch in flight) apart from the
// rest of the transfer (the staging, the copy's own submission and wait, and
// the host copy). A recorded upload flushes nothing, so its whole time is
// the rest. Off, the two methods read one atomic each and nothing else. A
// transfer that fails part-way records nothing.

/// The clock's words, in order: what each cumulative counter holds.
pub const TRANSFER_CLOCK_NAMES: [&str; 7] = [
    "download_calls",
    "download_flush_ns",
    "download_copy_ns",
    "upload_calls",
    "upload_flush_ns",
    "upload_copy_ns",
    "upload_recorded_calls",
];
static TRANSFER_CLOCK_ON: AtomicBool = AtomicBool::new(false);
static TRANSFER_CLOCK: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Switch the transfer clock on or off for this process.
pub fn set_transfer_clock(on: bool) {
    TRANSFER_CLOCK_ON.store(on, Ordering::SeqCst);
}

/// Whether the transfer clock is on.
pub fn transfer_clock_on() -> bool {
    TRANSFER_CLOCK_ON.load(Ordering::Relaxed)
}

/// The clock's cumulative words (`TRANSFER_CLOCK_NAMES`); a caller
/// differences two readings.
pub fn transfer_clock_ns() -> [u64; 7] {
    std::array::from_fn(|i| TRANSFER_CLOCK[i].load(Ordering::Relaxed))
}

/// Record one finished transfer: `calls` is word 0 (download) or 3 (upload),
/// with its flush and its rest in the two words after it.
fn transfer_clock_add(calls: usize, flush_ns: u64, rest_ns: u64) {
    TRANSFER_CLOCK[calls].fetch_add(1, Ordering::Relaxed);
    TRANSFER_CLOCK[calls + 1].fetch_add(flush_ns, Ordering::Relaxed);
    TRANSFER_CLOCK[calls + 2].fetch_add(rest_ns, Ordering::Relaxed);
}

/// Nanoseconds between two instants, saturating at u64::MAX.
fn clock_ns(from: std::time::Instant, to: std::time::Instant) -> u64 {
    u64::try_from(to.saturating_duration_since(from).as_nanos()).unwrap_or(u64::MAX)
}

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
    /// `shaderIntegerDotProduct` (core 1.3 / VK_KHR_shader_integer_dot_product):
    /// queried, not required -- `open` enables it when the driver has it, and
    /// nothing here fails without it. `decode_probe`'s VQ8 kernel declares
    /// `dotPacked4x8EXT` and reads this field to decide whether to dispatch it.
    pub integer_dot_product: bool,
    /// `shaderFloat16` (core 1.2): queried, not required -- `open` enables it
    /// when the driver has it. The fp16 cooperative-matrix modules declare the
    /// SPIR-V `Float16` capability, so `ack_matmul_f16` builds them only on a
    /// device where this is set (and `f16_cooperative_matrix` is too).
    pub shader_float16: bool,
    /// `shaderRoundingModeRTEFloat16` and `shaderDenormPreserveFloat16` (core
    /// 1.2 float controls, 2026-09-29): a module may declare round-to-nearest-
    /// even and denormal preservation for 16-bit floats, so a hardware
    /// conversion to binary16 is IEEE's (subnormals kept, an overflow to
    /// infinity). The fused attention's fp16 modules take the hardware
    /// conversion only when both are set (`ack_attention_f16_fwd`).
    pub f16_rounding_rte: bool,
    pub f16_denorm_preserve: bool,
    /// `shaderDenormPreserveFloat32` and `shaderSignedZeroInfNanPreserveFloat32`
    /// (core 1.2 float controls, 2026-10-03): a module may declare that its
    /// FP32 arithmetic keeps subnormals and keeps a zero's sign, an infinity
    /// and a NaN through every operation (no `0 + x -> x` folding). The MoE
    /// combine moe-combine/v1 declares both, because its sum (`out + w * y`
    /// from +0.0, the reference model's order) keeps subnormal products and
    /// turns `0 + -0` into `+0`; `MoeCombineKernel` refuses a device where
    /// either is unset.
    pub f32_denorm_preserve: bool,
    pub f32_signed_zero_inf_nan_preserve: bool,
    /// The 16x16x16 fp16 x fp16 -> f32 subgroup-scope cooperative shape the
    /// fp16 matmul modules are written for is among the driver's properties:
    /// the fp16 counterpart of `bf16_cooperative_matrix`, for a device such as
    /// NVIDIA Turing whose tensor cores take fp16 and not bf16.
    pub f16_cooperative_matrix: bool,
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
    /// R1: whether `Context::open` found a queue family that supports
    /// `TRANSFER` and neither `COMPUTE` nor `GRAPHICS` (a dedicated DMA
    /// family) and was able to request a queue from it. `false` on a device
    /// with no such family, or with only one queue family altogether --
    /// `copy_from_staging_async` still works on such a device (R2's
    /// fallback), it just does the ordinary synchronous copy on the compute
    /// queue instead of overlapping it.
    pub transfer_queue: bool,
    /// The family index `transfer_queue` was taken from, when it is `true`.
    pub transfer_queue_family: Option<u32>,
    /// R2: whether the device reports `VK_KHR_timeline_semaphore` (core in
    /// Vulkan 1.2) and `Context::open` enabled it. `false` makes
    /// `copy_from_staging_async` take the same synchronous fallback as a
    /// missing transfer queue.
    pub timeline_semaphores: bool,
    /// `VK_EXT_external_memory_host`: whether the driver reports it, in which
    /// case `open` enables it. It lets memory this process allocated itself be
    /// imported as device-visible memory with no copy. The streaming
    /// pipeline's pinned host ring imports its ring this way, so an unbuffered
    /// file read lands directly in memory the transfer queue copies from.
    /// Queried, not required: without it the ring falls back to a mapped
    /// HOST_CACHED staging area, and says so.
    pub external_memory_host: bool,
    /// `minImportedHostPointerAlignment` of that extension: the address and
    /// size alignment an import needs. `0` when the extension is absent.
    pub min_imported_host_pointer_alignment: u64,
    /// How `Context::open` chose this device: the `ACK_DEVICE` choice in
    /// force, the rule that picked it and any device it did not admit
    /// (`device_choice.rs`). Declared by the kit, not read from the driver.
    pub selection: String,
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

impl CoopmatShape {
    /// The shape every cooperative-matrix product here is written for:
    /// 16x16x16 at subgroup scope with `operand` A and B and an f32 result.
    pub fn is_16x16x16_subgroup_to_f32(&self, operand: &str) -> bool {
        self.m == 16
            && self.n == 16
            && self.k == 16
            && self.a_type == operand
            && self.b_type == operand
            && self.result_type == "float32"
            && self.scope == "subgroup"
    }
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

/// `(timeline semaphore, value)` pairs a submission waits on or signals.
type TimelineValues = Vec<(vk::Semaphore, u64)>;

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
    /// R1: a queue from a dedicated TRANSFER (DMA) family, when the device
    /// exposes one (`report.transfer_queue`). `None` on a device without one
    /// -- every method below that would use it falls back to the compute
    /// queue instead (R2's `copy_from_staging_async` fallback).
    transfer_queue: Option<vk::Queue>,
    /// The family `transfer_queue` was taken from; also `report.transfer_queue_family`.
    transfer_queue_family: Option<u32>,
    /// The transfer queue's own command pool (one pool per family). `None`
    /// exactly when `transfer_queue` is `None`. Created and destroyed
    /// alongside `command_pool`, outside `live_objects()`'s count.
    transfer_command_pool: Option<vk::CommandPool>,
    /// R2: one owned timeline semaphore, or `None` when the driver does not
    /// report `VK_KHR_timeline_semaphore` (`report.timeline_semaphores`).
    timeline: Option<vk::Semaphore>,
    /// The last value this context signalled `timeline` to; the next async
    /// copy signals `timeline_value + 1`. Meaningless when `timeline` is `None`.
    timeline_value: Cell<u64>,
    /// The largest ticket value `await_transfer` has armed and no submission
    /// has yet consumed (R2). Waiting for the largest value implies every
    /// smaller one has also been reached, since the timeline only counts up,
    /// so a single slot -- not a list -- is enough for any number of pending
    /// `await_transfer` calls between two submissions. Consumed (taken) by
    /// the next `one_shot` or `flush` submission on the compute queue, which
    /// is what "the next submission's wait list" means (R2's doc comment).
    pending_transfer_wait: Cell<Option<u64>>,
    /// Async transfer-queue command buffers not yet known to be safe to free
    /// (R2): `(ticket value, command buffer)`, reaped opportunistically by
    /// `copy_from_staging_async` and by `wait_transfer` once
    /// `vkGetSemaphoreCounterValue` shows the timeline has passed that value,
    /// and unconditionally at `Drop` (after `device_wait_idle`, which
    /// guarantees every one of them has by then). Each entry is a tracked
    /// object (`vk_allocate_transfer_command_buffer` calls `track(1)`), so
    /// `live_objects()` counts it exactly as long as it is live here.
    pending_transfer_cbs: RefCell<Vec<(u64, vk::CommandBuffer)>>,
    /// Timeline values the NEXT compute submission signals when it completes.
    /// This is COMPUTE_DONE: the streaming pipeline (`stream`) has its transfer
    /// queue wait on it before it overwrites a VRAM slot a kernel was reading.
    /// Consumed by whichever of `one_shot` and `flush` submits next, exactly
    /// like `pending_transfer_wait`. If that submission fails, the values are
    /// signalled from the host instead (`release_signals`), so a queue waiting
    /// on them is not left waiting forever.
    pending_signals: RefCell<Vec<(vk::Semaphore, u64)>>,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub report: DeviceReport,
    /// Vulkan 1.3 subgroupSizeControl + computeFullSubgroups were enabled.
    pub subgroup_size_control: bool,
    /// Vulkan 1.2 storageBuffer8BitAccess and shaderInt8 were both enabled,
    /// which the pointwise family's bool-storage modules declare; without
    /// them those modules are refused by name rather than built.
    pub bool_storage: bool,
    /// The device's [minSubgroupSize, maxSubgroupSize]: the only widths a
    /// compute pipeline may require (see `can_require_subgroup_size`).
    pub subgroup_size_range: (u32, u32),
    /// requiredSubgroupSizeStages includes COMPUTE.
    pub required_subgroup_size_compute: bool,
    /// The device reports shaderRoundingModeRTEFloat16/64 and
    /// shaderDenormPreserveFloat16/64 (core 1.2 float controls): the modes the
    /// fp16-output product modules declare, for 16- and 64-bit alike, since a
    /// device whose float controls are only 32-bit independent needs both.
    /// Without all four those modules are refused by code, not built.
    pub fp16_rounding_declarable: bool,
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
    /// Whether `dispatch` may reuse the descriptor set a kernel's CURRENT
    /// bindings were last written into, within the same batch generation
    /// (`ACK_DESCRIPTOR_REUSE`, default on; `0` turns it off for an A/B). The
    /// write it skips is identical, and the set is in the recording batch's own
    /// bank, so nothing in flight can see it change.
    descriptor_reuse: Cell<bool>,
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
    /// Fences of COMPLETED submissions, reset and kept for the next submission
    /// instead of destroyed (2026-09-22), while `fence_pool_enabled`. Measured
    /// on an NVIDIA L4 (driver 580.173.02) under the torch adapter: one A0
    /// training update made 2,186 `vkCreateFence` and 2,187 `vkDestroyFence`
    /// calls costing 2.03 s of host time between them, one pair per batch and
    /// per one-shot submission. A pooled fence is a LIVE object, counted by
    /// `live_objects()` and by `pooled_objects()`, as a pooled buffer is; it is
    /// in the unsignalled state (reset when it was pooled), and at most
    /// `batch::FENCE_POOL_MAX` are held.
    fence_pool: RefCell<Vec<vk::Fence>>,
    /// `ACK_FENCE_POOL=1` at open, or `set_fence_pool_enabled`. OFF by default
    /// in the crate for the same reason `deferred_upload` is: the crate's
    /// leak and poison suites pin "back to the baseline" after a submission,
    /// which a held fence would turn into baseline + 1. The torch adapter sets
    /// it before it opens.
    fence_pool_enabled: Cell<bool>,
    /// Staging pairs of COMPLETED transfers, kept mapped for the next transfer
    /// of their kind and size class instead of destroyed (see `StagingPair`),
    /// while `staging_pool_enabled`. Pooled pairs are LIVE objects, two each,
    /// counted by `live_objects()` and by `pooled_objects()`.
    staging_pool: RefCell<HashMap<(TransferKind, u64), Vec<StagingPair>>>,
    /// `ACK_STAGING_POOL=1` at open, or `set_staging_pool_enabled`. OFF by
    /// default in the crate for the reason `fence_pool_enabled` is; the torch
    /// adapter turns it on.
    staging_pool_enabled: Cell<bool>,
    staging_pool_bytes: Cell<u64>,
    /// `batch::STAGING_POOL_MAX_BYTES`, or `ACK_STAGING_POOL_MAX_BYTES`: a
    /// return that would cross it destroys the pair instead.
    staging_pool_max_bytes: Cell<u64>,
    staging_hits: Cell<u64>,
    staging_misses: Cell<u64>,
    staging_returns: Cell<u64>,
    staging_declined: Cell<u64>,
    staging_evicted: Cell<u64>,
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
    /// keyed by the size `raw_buffer` was called with: `Buffer::allocated`,
    /// the request's size class (`pool_class`) while the pool is on.
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
    ///
    /// ORDERED by allocated size (2026-10-01), so a request whose own class is
    /// empty can take the smallest larger pair (`pool_pick`).
    pool: RefCell<BTreeMap<u64, Vec<PooledBuffer>>>,
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
    /// Whether `VK_EXT_memory_budget` is enabled, so a pool miss can read how much device-local memory the OS still
    /// grants this process before it allocates (`device_local_room`).
    memory_budget: bool,
    /// Whether a request whose own size class is empty may take the smallest
    /// pooled pair of a LARGER class, at most `POOL_REUSE_RATIO` times its
    /// own (`ACK_BUFFER_POOL_LARGER`, on by default, or
    /// `set_buffer_pool_reuse_larger`).
    pool_reuse_larger: Cell<bool>,
    /// Hits served by such a larger pair; they are counted in `pool_hits` too.
    pool_hits_larger: Cell<u64>,
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
    /// The dispatch tag a caller set (`set_dispatch_tag`), 0 when none: the
    /// key the completion interval of each item recorded from then on is
    /// charged to. Read once per item, and only with the timestamps on.
    disp_tag: Cell<u32>,
    /// The timed items' completion intervals, summed per dispatch tag
    /// (2026-10-03). `dispatch_tag_times` says what they are and are not.
    tag_times: RefCell<BTreeMap<u32, DispatchTagTime>>,
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

/// Which transfer a staging pair serves. An upload's pair is `TRANSFER_SRC` in
/// write-combined `HOST_VISIBLE | HOST_COHERENT` memory and a readback's is
/// `TRANSFER_DST` in the cached type `readback_memory_preferences` picks, so a
/// pooled pair never serves the other kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum TransferKind {
    Upload,
    Readback,
}

/// A host-visible staging pair, MAPPED from its creation to its destruction.
///
/// Measured on an NVIDIA L4 (driver 580.173.02) under the torch adapter, one
/// A0 training update allocated and freed 612 of these, one per transfer:
/// `vkFreeMemory` alone took 526 ms of host time (0.86 ms a call, each one an
/// unmap and a driver round trip) and `vkAllocateMemory` most of 206 ms more.
/// With the staging pool on, a pair whose transfer COMPLETED is kept for the
/// next transfer of its kind and size class (see `staging_class`).
///
/// `capacity` is the pair's size: at least the bytes of any transfer it
/// serves, and only the first `bytes` of it are ever copied or read. `ptr` is
/// the address of the persistent mapping of the whole allocation.
struct StagingPair {
    kind: TransferKind,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    ptr: usize,
    /// `HOST_COHERENT`: a readback through a mapping that is not coherent
    /// invalidates it before the read
    coherent: bool,
}

/// The size class a pooled staging pair is made in: the next power of two, at
/// least a page. A pair is pooled only under its own class, so a transfer is
/// always served by a pair at least its size, and at most twice it.
fn staging_class(bytes: u64) -> u64 {
    bytes.next_power_of_two().max(4096)
}

/// The most staging one readback uses: a larger download runs in chunks of
/// this size through one pair (`Context::download_into`). It is the pool's
/// largest pair, so with the pool on that pair is pooled like any other.
pub const READBACK_CHUNK_BYTES: u64 = batch::STAGING_POOL_MAX_PAIR_BYTES;

/// Device-buffer requests up to this size keep an exact (word-rounded) pool
/// key: a class would round a few kilobytes and buy no reuse worth its memory.
pub const POOL_CLASS_FLOOR: u64 = 64 * 1024;

/// Size classes per power of two above `POOL_CLASS_FLOOR`: a class is at most
/// 1/8 = 12.5% larger than any request it serves.
pub const POOL_CLASSES_PER_OCTAVE: u64 = 8;

/// The buffer pool's key, and the size a pooled pair is allocated at, for a
/// word-rounded request of `allocated` bytes: exact up to `POOL_CLASS_FLOOR`,
/// and above it rounded up to one of `POOL_CLASSES_PER_OCTAVE` steps of its
/// power of two -- in (2^e, 2^(e+1)] the step is 2^(e-3). Never smaller than
/// the request, a multiple of 4, monotonic, and clamped to `limit`
/// (`maxStorageBufferRange`): a request the device can hold is never refused
/// because its class is larger.
pub fn pool_class(allocated: u64, limit: u64) -> u64 {
    if allocated <= POOL_CLASS_FLOOR {
        return allocated;
    }
    // (2^e, 2^(e+1)] holds `allocated`, and that octave's width 2^e is cut
    // into POOL_CLASSES_PER_OCTAVE (a power of two) steps
    let e = 63 - u64::from((allocated - 1).leading_zeros());
    let step = 1u64 << (e - u64::from(POOL_CLASSES_PER_OCTAVE.trailing_zeros()));
    let class = allocated.div_ceil(step).saturating_mul(step);
    if class > limit {
        allocated
    } else {
        class
    }
}

/// How much larger than a request's own class a pooled pair may be and still
/// serve it, when that class is empty (`ACK_BUFFER_POOL_LARGER`, 2026-10-01).
/// MEASURED on 2026-09-30 (the stage-E step, RX 9070 XT): a pool miss costs
/// about 0.77 ms, while the pool held about 3,700 idle pairs and missed about
/// 220 times a step. A larger pair costs only the memory it holds for one
/// tenant's life.
pub const POOL_REUSE_RATIO: u64 = 2;

/// Device-local bytes a pool miss keeps free under the OS budget (`Context::device_local_room`): retired buffers
/// awaiting a flush and the driver's own allocations land there too.
pub const POOL_BUDGET_MARGIN: u64 = 256 << 20;

/// The pool key a request of `allocated` bytes (already its class) takes: its
/// own class if the pool holds a pair of it, else -- when `larger` -- the
/// smallest held class above it and at most `POOL_REUSE_RATIO` times it. At
/// or below `POOL_CLASS_FLOOR` the key stays exact. The map never holds an
/// empty list: every take and eviction removes the key it empties.
fn pool_pick<V>(pool: &BTreeMap<u64, V>, allocated: u64, larger: bool) -> Option<u64> {
    if pool.contains_key(&allocated) {
        return Some(allocated);
    }
    if !larger || allocated <= POOL_CLASS_FLOOR {
        return None;
    }
    let limit = allocated.saturating_mul(POOL_REUSE_RATIO);
    pool.range((Bound::Excluded(allocated), Bound::Included(limit)))
        .next()
        .map(|(k, _)| *k)
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
    /// The dispatch tag of every item recorded with the instrument on, in
    /// recording order: `tags[i]` is the tag of the item whose completion is
    /// stamp `i + 1` (an item past the query capacity keeps its tag here and
    /// has no stamp). Empty when the instrument is off.
    tags: Vec<u32>,
    /// Staging pairs of the uploads recorded into this batch (2026-09-10):
    /// host-visible memory, released when the batch's fence signals -- to the
    /// staging pool when it is on and the wait succeeded, destroyed otherwise.
    retired_staging: Vec<StagingPair>,
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
    /// `Recorder::tags`, carried to the readback that charges each interval.
    tags: Vec<u32>,
    timing_started: std::time::Instant,
    /// The descriptor bank its kernels' sets came from; the next batch takes the other.
    bank: usize,
    retired_staging: Vec<StagingPair>,
}

/// `raw_buffer`'s sharing-mode choice (R1). `Exclusive` is every buffer's
/// mode before this change, and stays the mode for the private staging
/// buffers `upload`, `upload_recorded` and `download` allocate for
/// themselves: the transfer queue never touches them, so there is nothing to
/// share. `Concurrent` is for a buffer the transfer queue may touch while the
/// compute queue also reads or writes it -- `Context::buffer`'s device-local
/// storage buffers (a possible `dst` of `copy_from_staging_async`) and
/// `Context::staging`'s mapped areas (a possible `src`, and R3's bindable
/// operand), both only when the device has a transfer queue; with none, they
/// stay `Exclusive` too, since only one queue ever touches them.
///
/// CONCURRENT IS THE MEASURED TRADE THIS FIRST CUT ACCEPTS, NOT A FREE WIN:
/// the spec for this tranche notes AMD's driver disables some buffer
/// compression for a concurrently-shared allocation. The alternative --
/// `EXCLUSIVE` plus an explicit queue-family-ownership-transfer barrier pair
/// around every copy -- was left for a later cut; whether the compression
/// this gives up costs more than the overlap the tranche buys back is for the
/// probe to measure, not this comment to assert.
#[derive(Clone, Copy)]
enum BufferSharing<'a> {
    Exclusive,
    Concurrent(&'a [u32]),
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
    /// Open the device `ACK_DEVICE` chooses (`device_choice.rs`), with the
    /// features the kernels need enabled. By default that is the best-ranked
    /// GPU: a discrete GPU first, then one with VK_KHR_cooperative_matrix,
    /// then one with VK_KHR_shader_bfloat16. A CPU implementation of Vulkan
    /// (llvmpipe, SwiftShader) opens only when the caller asks for it
    /// (`ACK_DEVICE=cpu` or `any`); a host with no GPU driver is refused by
    /// name rather than run on a CPU emulating a GPU. `ACK_DEVICE=gpu:<n>`
    /// takes the n-th GPU instead of the best-ranked one.
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

        // Which device: ACK_DEVICE admits GPUs only unless the caller names a
        // CPU implementation, and ranks what it admits (device_choice.rs).
        let choice = device_choice::DeviceChoice::from_env().map_err(AckError::NoDevice)?;
        let mut candidates = Vec::with_capacity(physicals.len());
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
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            candidates.push(device_choice::Candidate {
                name: props
                    .device_name_as_c_str()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                device_type: props.device_type,
                compute: families
                    .iter()
                    .any(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)),
                cooperative_matrix: has("VK_KHR_cooperative_matrix"),
                bfloat16: has("VK_KHR_shader_bfloat16"),
            });
        }
        let chosen = device_choice::choose(&candidates, choice).map_err(AckError::NoDevice)?;
        let physical = physicals[chosen.index];

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
        let has_external_memory_host = has_ext("VK_EXT_external_memory_host");
        let has_memory_budget = has_ext("VK_EXT_memory_budget");

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
        // The widths a pipeline may REQUIRE (core 1.3, like `v13_q` below):
        // `kernel_with_subgroup` refuses any other, because a driver is not
        // obliged to reject one and a kernel written for 64 lanes runs wrong,
        // not slow, at 32 (kernels/coopmat_bf16_tiled_v4.comp).
        let mut size_control_q = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        // core 1.2: which rounding and denormal modes a module may declare (the
        // fp16-output products' fp16_rounding_declarable and the report's
        // f16_rounding_rte / f16_denorm_preserve both read it)
        let mut float_controls_q = vk::PhysicalDeviceFloatControlsProperties::default();
        // VK_EXT_external_memory_host's import alignment, read only when the
        // driver reports the extension (chaining an unknown struct is invalid)
        let mut external_host_q = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut p2_q = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_q)
            .push_next(&mut driver_q)
            .push_next(&mut id_q)
            .push_next(&mut size_control_q)
            .push_next(&mut float_controls_q);
        if has_external_memory_host {
            p2_q = p2_q.push_next(&mut external_host_q);
        }
        unsafe { instance.get_physical_device_properties2(physical, &mut p2_q) };
        let fp16_rounding_declarable = float_controls_q.shader_rounding_mode_rte_float16
            == vk::TRUE
            && float_controls_q.shader_rounding_mode_rte_float64 == vk::TRUE
            && float_controls_q.shader_denorm_preserve_float16 == vk::TRUE
            && float_controls_q.shader_denorm_preserve_float64 == vk::TRUE;
        let subgroup_size_range = (
            size_control_q.min_subgroup_size,
            size_control_q.max_subgroup_size,
        );
        let required_subgroup_size_compute = size_control_q
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE);

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

        // Queried here, ahead of `report`, so `DeviceReport` records what the
        // driver said about shaderIntegerDotProduct (core 1.3 /
        // VK_KHR_shader_integer_dot_product) whether or not anything below
        // ever requests it. `v13_q` is reused below for subgroupSizeControl.
        let mut v13_q = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2_13 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13_q);
        unsafe { instance.get_physical_device_features2(physical, &mut f2_13) };

        let api = props.api_version;
        let f16_cooperative_matrix = coopmat_shapes
            .iter()
            .any(|s| s.is_16x16x16_subgroup_to_f32("float16"));
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
            integer_dot_product: v13_q.shader_integer_dot_product == vk::TRUE,
            // enabled at device creation exactly when the driver has it (below)
            shader_float16: v12_q.shader_float16 == vk::TRUE,
            f16_rounding_rte: float_controls_q.shader_rounding_mode_rte_float16 == vk::TRUE,
            f16_denorm_preserve: float_controls_q.shader_denorm_preserve_float16 == vk::TRUE,
            f32_denorm_preserve: float_controls_q.shader_denorm_preserve_float32 == vk::TRUE,
            f32_signed_zero_inf_nan_preserve: float_controls_q
                .shader_signed_zero_inf_nan_preserve_float32
                == vk::TRUE,
            f16_cooperative_matrix,
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
            // filled in below, once the queue families are inspected (R1/R2)
            transfer_queue: false,
            transfer_queue_family: None,
            timeline_semaphores: v12_q.timeline_semaphore == vk::TRUE,
            external_memory_host: has_external_memory_host,
            min_imported_host_pointer_alignment: if has_external_memory_host {
                external_host_q.min_imported_host_pointer_alignment
            } else {
                0
            },
            selection: chosen.selection,
        };
        let mut report = report;

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

        // R1: a dedicated transfer (DMA) queue, if the device exposes one --
        // TRANSFER support and NEITHER COMPUTE NOR GRAPHICS. `vulkaninfo` on
        // the device this was measured against reports exactly one such
        // family (index 2, `queueCount` 1). A device with no such family, or
        // with only one family altogether (so the loop below simply finds
        // nothing distinct from `queue_family`), carries on with the compute
        // queue alone; `copy_from_staging_async`'s fallback (R2) is what
        // makes that safe for a caller to not special-case.
        let transfer_family = families.iter().enumerate().find_map(|(i, f)| {
            let transfer = f.queue_flags.contains(vk::QueueFlags::TRANSFER);
            let compute = f.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let graphics = f.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            (transfer && !compute && !graphics).then_some(i as u32)
        });
        report.transfer_queue = transfer_family.is_some();
        report.transfer_queue_family = transfer_family;
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
            // schema-4 pointwise bool storage: the u8 modules declare
            // StorageBuffer8BitAccess and Int8, so both are enabled when the
            // driver reports them and `bool_storage` says whether it did
            .storage_buffer8_bit_access(v12_q.storage_buffer8_bit_access == vk::TRUE)
            .vulkan_memory_model(v12_q.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(v12_q.vulkan_memory_model_device_scope == vk::TRUE)
            .host_query_reset(v12_q.host_query_reset == vk::TRUE)
            // R2: enabled whenever the driver reports it, whether or not this
            // device also has a transfer queue -- `copy_from_staging_async`
            // reads `report.timeline_semaphores` (not the transfer queue
            // alone) to decide whether it can skip the synchronous fallback.
            .timeline_semaphore(report.timeline_semaphores);
        let mut v11_on =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        // Subgroup size control (core 1.3): tiled kernels are written for one
        // subgroup width and ask for it explicitly rather than trusting a heuristic.
        // (`v13_q` was queried above, ahead of `report`.)
        let subgroup_size_control =
            v13_q.subgroup_size_control == vk::TRUE && v13_q.compute_full_subgroups == vk::TRUE;
        let bool_storage =
            v12_q.storage_buffer8_bit_access == vk::TRUE && v12_q.shader_int8 == vk::TRUE;
        let mut v13_on = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(subgroup_size_control)
            .compute_full_subgroups(subgroup_size_control)
            .shader_integer_dot_product(v13_q.shader_integer_dot_product == vk::TRUE);
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
        let mut qci = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        // R1: a second queue, from the dedicated transfer family, requested
        // alongside the compute one so the two can run concurrently. Distinct
        // from `queue_family` by construction (the family search above
        // excludes COMPUTE), so this can never duplicate the first entry.
        if let Some(tf) = transfer_family {
            qci.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(tf)
                    .queue_priorities(&priorities),
            );
        }
        let mut ext_names: Vec<*const i8> = Vec::new();
        let coop_name = c"VK_KHR_cooperative_matrix";
        let bf16_name = c"VK_KHR_shader_bfloat16";
        if has_coopmat {
            ext_names.push(coop_name.as_ptr());
        }
        if has_bf16 {
            ext_names.push(bf16_name.as_ptr());
        }
        // the streaming pipeline's zero-copy pinned ring (see
        // `DeviceReport::external_memory_host`); enabling it changes nothing
        // for a context that never imports host memory
        let external_host_name = c"VK_EXT_external_memory_host";
        if has_external_memory_host {
            ext_names.push(external_host_name.as_ptr());
        }
        // the buffer pool's budget check (`device_local_room`)
        let memory_budget_name = c"VK_EXT_memory_budget";
        if has_memory_budget {
            ext_names.push(memory_budget_name.as_ptr());
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

        // R1: the transfer queue's own command pool, bound to its family
        // (a command pool is created for exactly one family; the compute
        // pool above cannot record work for this queue). Created here,
        // alongside `command_pool`, so it is one of "open's own objects" and
        // outside `live_objects()`'s count -- exactly like `command_pool`
        // itself (see `live_objects`'s doc comment).
        let transfer_queue = transfer_family.map(|tf| unsafe { device.get_device_queue(tf, 0) });
        let transfer_command_pool = match transfer_family {
            Some(tf) => {
                let tp_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(tf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                match unsafe { device.create_command_pool(&tp_info, None) } {
                    Ok(pool) => Some(pool),
                    Err(e) => {
                        unsafe {
                            device.destroy_command_pool(command_pool, None);
                            device.destroy_device(None);
                        }
                        return Err(e.into());
                    }
                }
            }
            None => None,
        };

        // R2: one owned timeline semaphore, its counter starting at 0 (no
        // copy has ever signalled it). `SemaphoreTypeCreateInfo` chains into
        // the ordinary `SemaphoreCreateInfo` the same way every other
        // extension struct in this file chains into its root.
        let timeline = if report.timeline_semaphores {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            match unsafe { device.create_semaphore(&sem_info, None) } {
                Ok(s) => Some(s),
                Err(e) => {
                    unsafe {
                        if let Some(tp) = transfer_command_pool {
                            device.destroy_command_pool(tp, None);
                        }
                        device.destroy_command_pool(command_pool, None);
                        device.destroy_device(None);
                    }
                    return Err(e.into());
                }
            }
        } else {
            None
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
            transfer_queue,
            transfer_queue_family: transfer_family,
            transfer_command_pool,
            timeline,
            timeline_value: Cell::new(0),
            pending_transfer_wait: Cell::new(None),
            pending_transfer_cbs: RefCell::new(Vec::new()),
            pending_signals: RefCell::new(Vec::new()),
            memory,
            report,
            subgroup_size_control,
            bool_storage,
            subgroup_size_range,
            required_subgroup_size_compute,
            fp16_rounding_declarable,
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
            descriptor_reuse: Cell::new(batch::env_flag("ACK_DESCRIPTOR_REUSE", true)),
            bank: Cell::new(0),
            in_flight: RefCell::new(None),
            async_flush: Cell::new(batch::env_flag("ACK_ASYNC_FLUSH", true)),
            ramp: Cell::new(0),
            batch_ramp: Cell::new(batch::env_flag("ACK_BATCH_RAMP", true)),
            deferred_upload: Cell::new(batch::env_flag("ACK_DEFERRED_UPLOAD", false)),
            fence_pool: RefCell::new(Vec::new()),
            fence_pool_enabled: Cell::new(batch::env_flag("ACK_FENCE_POOL", false)),
            staging_pool: RefCell::new(HashMap::new()),
            staging_pool_enabled: Cell::new(batch::env_flag("ACK_STAGING_POOL", false)),
            staging_pool_bytes: Cell::new(0),
            staging_pool_max_bytes: Cell::new(batch::env_u64(
                "ACK_STAGING_POOL_MAX_BYTES",
                batch::STAGING_POOL_MAX_BYTES,
            )),
            staging_hits: Cell::new(0),
            staging_misses: Cell::new(0),
            staging_returns: Cell::new(0),
            staging_declined: Cell::new(0),
            staging_evicted: Cell::new(0),
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
            pool: RefCell::new(BTreeMap::new()),
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
            //
            // WITH `VK_EXT_memory_budget` THE CEILING IS THE WHOLE HEAP (2026-10-21): a pool miss then gives memory
            // back before the OS budget is crossed (`device_local_room`), so the ceiling no longer has to leave room
            // for live tensors. Measured on the RX 9070 XT, a 26M-parameter training step at microbatch 32 (live
            // peak 12.3 GB): half the heap declined ~200 returns a step and ran 30-37k tokens/s with 13 s spikes; the
            // whole heap ran a steady 52.5k, and microbatch 16 was unchanged (89-90k).
            pool_max_bytes: Cell::new(batch::env_u64(
                "ACK_BUFFER_POOL_BYTES",
                if has_memory_budget && batch::env_flag("ACK_POOL_BUDGET", true) {
                    device_local_bytes
                } else {
                    device_local_bytes / 2
                }
                .max(64 << 20),
            )),
            pool_enabled: Cell::new(batch::env_flag("ACK_BUFFER_POOL", true)),
            memory_budget: has_memory_budget && batch::env_flag("ACK_POOL_BUDGET", true),
            pool_reuse_larger: Cell::new(batch::env_flag("ACK_BUFFER_POOL_LARGER", true)),
            pool_hits_larger: Cell::new(0),
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
            disp_tag: Cell::new(0),
            tag_times: RefCell::new(BTreeMap::new()),
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

    /// Let a request whose own size class is empty take the smallest pooled
    /// pair of a larger class, at most `POOL_REUSE_RATIO` times its own
    /// (`pool_pick`), or keep every request to its own class.
    pub fn set_buffer_pool_reuse_larger(&self, on: bool) {
        self.pool_reuse_larger.set(on);
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
            hits_larger: self.pool_hits_larger.get(),
            misses: self.pool_misses.get(),
            returns: self.pool_returns.get(),
            declined: self.pool_declined.get(),
            drained: self.pool_drained.get(),
            reclaims: self.pool_reclaims.get(),
            held_objects: self.pooled_buffer_objects(),
            held_bytes: self.pool_bytes.get(),
        }
    }

    /// Vulkan objects the pools hold: two per buffer pair, one per pooled
    /// fence, two per pooled staging pair. They are LIVE objects and `live_objects()` counts them, because
    /// they are: the pools are holding them, not leaking them. A leak assertion
    /// that wants the un-pooled quantity subtracts this, or runs on a context
    /// with the pools off.
    pub fn pooled_objects(&self) -> i64 {
        self.pooled_buffer_objects() + self.pooled_fences() + 2 * self.pooled_staging()
    }

    /// Objects the dispatch timestamp instrument holds for the context's
    /// lifetime: its query pool, created the first time the instrument is on
    /// and destroyed with the context. Live, not pooled, and not a leak. At
    /// quiescence the leak law of a context that used the instrument is
    /// `live_objects() == pooled_objects() + instrument_objects()` (window
    /// 18f0d9da's probes reported this one object as a leak).
    pub fn instrument_objects(&self) -> i64 {
        i64::from(self.ts_pool.get() != vk::QueryPool::null())
    }

    /// The buffer pool's share of `pooled_objects()`: the objects that hold
    /// device memory, which is what the allocation retry and `pool_stats()`
    /// are about.
    fn pooled_buffer_objects(&self) -> i64 {
        self.pool
            .borrow()
            .values()
            .map(|v| v.len() as i64 * 2)
            .sum()
    }

    /// Take a pair for a request of `allocated` bytes (already its class):
    /// one of its own class, or -- when larger-class reuse is on and its own
    /// class is empty -- the smallest larger pair within `POOL_REUSE_RATIO`
    /// (`pool_pick`). The pair keeps its own `allocated`, so it goes back to
    /// its own class when it is freed.
    fn take_pooled(&self, allocated: u64) -> Option<PooledBuffer> {
        if !self.pool_enabled.get() {
            return None;
        }
        let mut pool = self.pool.borrow_mut();
        let key = pool_pick(&pool, allocated, self.pool_reuse_larger.get())?;
        let entry = pool.get_mut(&key)?;
        let p = entry.pop()?;
        if entry.is_empty() {
            pool.remove(&key);
        }
        if key != allocated {
            self.pool_hits_larger.set(self.pool_hits_larger.get() + 1);
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
        let held: Vec<PooledBuffer> = std::mem::take(&mut *self.pool.borrow_mut())
            .into_values()
            .flatten()
            .collect();
        for p in held {
            self.vk_destroy_buffer(p.buffer);
            self.vk_free_memory(p.memory);
            self.pool_drained.set(self.pool_drained.get() + 1);
        }
        self.pool_bytes.set(0);
    }

    /// Destroy held pairs, the largest first, until at least `want` bytes are
    /// freed or the pool is empty; returns the bytes freed. The allocation
    /// retry's reclaim (`Context::buffer`): the largest pairs free the most
    /// memory per destroy, and the small ones the next requests will take stay.
    fn evict_largest(&self, want: u64) -> u64 {
        let mut freed = 0u64;
        while freed < want {
            let taken = {
                let mut pool = self.pool.borrow_mut();
                let key = match pool.keys().next_back() {
                    Some(k) => *k,
                    None => break,
                };
                let entry = pool.get_mut(&key).expect("key just read");
                let p = entry.pop();
                if entry.is_empty() {
                    pool.remove(&key);
                }
                p
            };
            let Some(p) = taken else { break };
            self.pool_bytes
                .set(self.pool_bytes.get().saturating_sub(p.allocated));
            self.vk_destroy_buffer(p.buffer);
            self.vk_free_memory(p.memory);
            self.pool_drained.set(self.pool_drained.get() + 1);
            freed += p.allocated;
        }
        freed
    }

    /// Device-local bytes the OS still grants this process on the largest DEVICE_LOCAL heap (`heapBudget -
    /// heapUsage`, `VK_EXT_memory_budget`), or `None` without the extension (or with `ACK_POOL_BUDGET=0`).
    fn device_local_room(&self) -> Option<u64> {
        if !self.memory_budget {
            return None;
        }
        let n = self.memory.memory_heap_count as usize;
        let heap = (0..n)
            .filter(|&i| {
                self.memory.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .max_by_key(|&i| self.memory.memory_heaps[i].size)?;
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                self.instance
                    .get_physical_device_memory_properties2(self.physical, &mut props)
            };
        }
        Some(budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap]))
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

    /// Take the wait `await_transfer` armed, if any, pairing it with the
    /// timeline semaphore itself (R2). `None` on a context with no timeline
    /// semaphore, even if a value were somehow armed (`await_transfer`
    /// refuses to arm one on such a context, so this is belt and braces, not
    /// a path that fires). Consumes the armed value: only the caller's own
    /// submission carries it.
    fn take_pending_transfer_wait(&self) -> Option<(vk::Semaphore, u64)> {
        let value = self.pending_transfer_wait.take()?;
        self.timeline.map(|s| (s, value))
    }

    /// Make the next compute submission signal `semaphore` (a timeline) to
    /// `value` when it completes. Crate-internal: COMPUTE_DONE.
    pub(crate) fn arm_submission_signal(&self, semaphore: vk::Semaphore, value: u64) {
        self.pending_signals.borrow_mut().push((semaphore, value));
    }

    /// Whether a signal is armed and not yet carried by a submission.
    pub(crate) fn has_armed_signals(&self) -> bool {
        !self.pending_signals.borrow().is_empty()
    }

    /// Drop every armed, not yet submitted signal of `semaphore`: its owner
    /// is about to destroy it, and no later submission may name it.
    pub(crate) fn disarm_signals(&self, semaphore: vk::Semaphore) {
        self.pending_signals
            .borrow_mut()
            .retain(|&(s, _)| s != semaphore);
    }

    /// Everything the next compute submission waits on and signals: the
    /// transfer-timeline wait `await_transfer` armed, and the signals
    /// `arm_submission_signal` armed. Consumes both. With nothing armed, both
    /// lists are empty and the submit info the callers build is the one they
    /// built before this hook.
    fn take_submission_sync(&self) -> (TimelineValues, TimelineValues) {
        let waits: TimelineValues = self.take_pending_transfer_wait().into_iter().collect();
        let signals = std::mem::take(&mut *self.pending_signals.borrow_mut());
        (waits, signals)
    }

    /// A submission that carried `signals` failed before the queue accepted
    /// it, so the device will never signal them. Signal them from the host,
    /// best effort, so a queue waiting on one is released rather than hung.
    /// That matters at `Drop`, whose `device_wait_idle` would otherwise wait
    /// forever. A failure here (a lost device) is ignored: a lost device's
    /// queues do not wait.
    fn release_signals(&self, signals: &[(vk::Semaphore, u64)]) {
        for &(semaphore, value) in signals {
            let current =
                unsafe { self.device.get_semaphore_counter_value(semaphore) }.unwrap_or(u64::MAX);
            if current < value {
                let info = vk::SemaphoreSignalInfo::default()
                    .semaphore(semaphore)
                    .value(value);
                let _ = unsafe { self.device.signal_semaphore(&info) };
            }
        }
    }

    /// Release a submission's fence and command buffer and count the release
    /// in the same place, so the count cannot say "released" when nothing was.
    fn release_submission(&self, fence: vk::Fence, free_cb: impl FnOnce()) {
        self.vk_destroy_fence(fence);
        free_cb();
        self.torn_down.set(self.torn_down.get() + 1);
    }

    /// Release a submission whose fence WAIT SUCCEEDED: the fence has
    /// signalled, so nothing can still be using it, and with the fence pool on
    /// it is reset and kept rather than destroyed. Every other release (a
    /// failed wait, a lost device, a submission that never reached the queue)
    /// takes `release_submission`, because only a completed wait establishes
    /// that the fence is free to reset. A reset that fails, or a full pool,
    /// destroys the fence as before.
    fn release_completed_submission(&self, fence: vk::Fence, free_cb: impl FnOnce()) {
        let pooled = self.fence_pool_enabled.get()
            && self.fence_pool.borrow().len() < batch::FENCE_POOL_MAX
            && unsafe { self.device.reset_fences(&[fence]) }.is_ok();
        if !pooled {
            self.release_submission(fence, free_cb);
            return;
        }
        self.fence_pool.borrow_mut().push(fence);
        free_cb();
        self.torn_down.set(self.torn_down.get() + 1);
    }

    /// Whether completed submissions' fences are pooled (see `fence_pool`).
    pub fn fence_pool_enabled(&self) -> bool {
        self.fence_pool_enabled.get()
    }

    /// Turn the fence pool on or off, as `ACK_FENCE_POOL` does at open.
    /// Turning it off destroys the fences it holds.
    pub fn set_fence_pool_enabled(&self, on: bool) {
        self.fence_pool_enabled.set(on);
        if !on {
            self.drain_fence_pool();
        }
    }

    /// Fences the pool holds. They are live objects: `live_objects()` and
    /// `pooled_objects()` both count them.
    pub fn pooled_fences(&self) -> i64 {
        self.fence_pool.borrow().len() as i64
    }

    fn drain_fence_pool(&self) {
        let held: Vec<vk::Fence> = self.fence_pool.borrow_mut().drain(..).collect();
        for fence in held {
            self.vk_destroy_fence(fence);
        }
    }

    /// Whether completed transfers' staging pairs are pooled (see `StagingPair`).
    pub fn staging_pool_enabled(&self) -> bool {
        self.staging_pool_enabled.get()
    }

    /// Turn the staging pool on or off, as `ACK_STAGING_POOL` does at open.
    /// Turning it off destroys the pairs it holds.
    pub fn set_staging_pool_enabled(&self, on: bool) {
        self.staging_pool_enabled.set(on);
        if !on {
            self.drain_staging_pool();
        }
    }

    /// Ceiling on the bytes the staging pool holds; the pairs above it are
    /// destroyed at once.
    pub fn set_staging_pool_max_bytes(&self, bytes: u64) {
        self.staging_pool_max_bytes.set(bytes);
        if self.staging_pool_bytes.get() > bytes {
            self.drain_staging_pool();
        }
    }

    /// Staging pairs the pool holds. Each is two live objects: `live_objects()`
    /// counts them and `pooled_objects()` names them.
    pub fn pooled_staging(&self) -> i64 {
        self.staging_pool
            .borrow()
            .values()
            .map(|v| v.len() as i64)
            .sum()
    }

    /// Transfers served from the staging pool, transfers that had to create
    /// their pair, pairs the pool took back, pairs it declined (its ceiling,
    /// or a full class), and what it holds.
    ///
    /// ASSERT THESE ALONGSIDE THE BYTES, as for the buffer pool: a transfer
    /// through a reused pair and one through a fresh pair return the same bytes.
    pub fn staging_pool_stats(&self) -> StagingPoolStats {
        StagingPoolStats {
            hits: self.staging_hits.get(),
            misses: self.staging_misses.get(),
            returns: self.staging_returns.get(),
            declined: self.staging_declined.get(),
            evicted: self.staging_evicted.get(),
            held_pairs: self.pooled_staging(),
            held_bytes: self.staging_pool_bytes.get(),
        }
    }

    fn drain_staging_pool(&self) {
        let held: Vec<StagingPair> = self
            .staging_pool
            .borrow_mut()
            .drain()
            .flat_map(|(_, v)| v)
            .collect();
        self.staging_pool_bytes.set(0);
        for pair in held {
            self.destroy_staging_pair(pair);
        }
    }

    /// A mapped staging pair of at least `bytes` for `kind`: from the pool when
    /// it is on and holds one of the class, created otherwise -- in the class
    /// size with the pool on, so it can be pooled when its transfer completes,
    /// and in exactly `bytes` with it off, as before the pool existed.
    fn take_staging(&self, kind: TransferKind, bytes: u64) -> Result<StagingPair> {
        let pooling = self.staging_pool_enabled.get();
        if pooling {
            let class = staging_class(bytes);
            let hit = self
                .staging_pool
                .borrow_mut()
                .get_mut(&(kind, class))
                .and_then(|v| v.pop());
            if let Some(pair) = hit {
                self.staging_pool_bytes
                    .set(self.staging_pool_bytes.get() - pair.capacity);
                self.staging_hits.set(self.staging_hits.get() + 1);
                return Ok(pair);
            }
            self.staging_misses.set(self.staging_misses.get() + 1);
        }
        let capacity = if pooling { staging_class(bytes) } else { bytes };
        match self.create_staging(kind, capacity) {
            Ok(pair) => Ok(pair),
            // THE POOL MUST NEVER BE THE REASON A TRANSFER COULD NOT BE STAGED:
            // it holds host-visible memory nothing owns, so a failure with a
            // non-empty pool gives that back and tries once more
            Err(e) if self.pooled_staging() > 0 => {
                let _ = e;
                self.drain_staging_pool();
                self.create_staging(kind, capacity)
            }
            Err(e) => Err(e),
        }
    }

    /// Create and map a staging pair of `capacity` bytes for `kind`. Every
    /// failure destroys what the call made (ACK-L0-05).
    fn create_staging(&self, kind: TransferKind, capacity: u64) -> Result<StagingPair> {
        let (buffer, memory, flags) = match kind {
            // Exclusive, unconditionally: a staging pair is only ever named by
            // the one transfer that holds it, on the compute queue
            TransferKind::Upload => {
                let flags =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let (b, m) = self.raw_buffer(
                    capacity,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    flags,
                    BufferSharing::Exclusive,
                )?;
                (b, m, flags)
            }
            TransferKind::Readback => {
                // Preference order, best first (see `readback_memory_preferences`)
                let mut chosen = None;
                let mut last_err = None;
                for flags in Self::readback_memory_preferences() {
                    match self.raw_buffer(
                        capacity,
                        vk::BufferUsageFlags::TRANSFER_DST,
                        flags,
                        BufferSharing::Exclusive,
                    ) {
                        Ok((b, m)) => {
                            chosen = Some((b, m, flags));
                            break;
                        }
                        Err(e) => last_err = Some(e),
                    }
                }
                match chosen {
                    Some(t) => t,
                    None => {
                        return Err(last_err.unwrap_or_else(|| {
                            AckError::Unsupported("no host-visible memory type for readback".into())
                        }))
                    }
                }
            }
        };
        let ptr = match unsafe {
            self.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as usize,
            Err(e) => {
                self.vk_destroy_buffer(buffer);
                self.vk_free_memory(memory);
                return Err(e.into());
            }
        };
        Ok(StagingPair {
            kind,
            buffer,
            memory,
            capacity,
            ptr,
            coherent: flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
        })
    }

    fn destroy_staging_pair(&self, pair: StagingPair) {
        unsafe { self.device.unmap_memory(pair.memory) };
        self.vk_destroy_buffer(pair.buffer);
        self.vk_free_memory(pair.memory);
    }

    /// Hand back a staging pair NO SUBMISSION CAN STILL NAME: its transfer's
    /// fence wait succeeded, or the command that named it was never submitted.
    /// Pooled when the pool is on, the context is healthy, the pair is in its
    /// own size class (a pair made with the pool off is exactly its transfer's
    /// size and is not), it is at most `batch::STAGING_POOL_MAX_PAIR_BYTES`,
    /// and its class holds fewer than `batch::STAGING_POOL_MAX_PER_CLASS`;
    /// destroyed otherwise. Every other release -- a failed wait, a lost
    /// device, a poisoned context -- goes through `release_staging` as before.
    ///
    /// A RETURN THAT WOULD CROSS THE BYTE CEILING EVICTS LARGER PAIRS FIRST.
    /// Measured on an NVIDIA L4 through the torch adapter (2026-09-22): the
    /// model's parameter upload at the start of a run returned megabyte-class
    /// pairs until the ceiling was full, no training transfer ever took one of
    /// those classes again, and every later return -- the 32 KiB index
    /// readbacks, the few-KiB canonical images -- was DECLINED: 1,418 pairs
    /// created and destroyed in a two-update run with the pool on, exactly as
    /// with it off. A pool that holds what nothing asks for must give way to
    /// what is asked for; a return larger than everything the pool holds is
    /// still declined, so the small working set is never evicted for it.
    fn recycle_staging(&self, pair: StagingPair) {
        let class_len = |p: &StagingPair| {
            self.staging_pool
                .borrow()
                .get(&(p.kind, p.capacity))
                .map_or(0, |v| v.len())
        };
        let poolable = self.staging_pool_enabled.get()
            && !self.is_poisoned()
            && pair.capacity == staging_class(pair.capacity)
            && pair.capacity <= batch::STAGING_POOL_MAX_PAIR_BYTES
            && pair.capacity <= self.staging_pool_max_bytes.get()
            && class_len(&pair) < batch::STAGING_POOL_MAX_PER_CLASS;
        if poolable {
            while self.staging_pool_bytes.get() + pair.capacity > self.staging_pool_max_bytes.get()
                && self.evict_one_larger_staging(pair.capacity)
            {}
        }
        let pooled = poolable
            && self.staging_pool_bytes.get() + pair.capacity <= self.staging_pool_max_bytes.get();
        if !pooled {
            if self.staging_pool_enabled.get() {
                self.staging_declined.set(self.staging_declined.get() + 1);
            }
            self.destroy_staging_pair(pair);
            return;
        }
        self.staging_pool_bytes
            .set(self.staging_pool_bytes.get() + pair.capacity);
        self.staging_returns.set(self.staging_returns.get() + 1);
        self.staging_pool
            .borrow_mut()
            .entry((pair.kind, pair.capacity))
            .or_default()
            .push(pair);
    }

    /// Destroy one pooled pair of the largest class above `than` bytes, if the
    /// pool holds one: the eviction `recycle_staging` makes room with.
    fn evict_one_larger_staging(&self, than: u64) -> bool {
        let victim = {
            let mut pool = self.staging_pool.borrow_mut();
            let key = pool
                .iter()
                .filter(|(k, v)| k.1 > than && !v.is_empty())
                .map(|(k, _)| *k)
                .max_by_key(|k| k.1);
            key.and_then(|k| pool.get_mut(&k).and_then(|v| v.pop()))
        };
        match victim {
            Some(p) => {
                self.staging_pool_bytes
                    .set(self.staging_pool_bytes.get() - p.capacity);
                self.staging_evicted.set(self.staging_evicted.get() + 1);
                self.destroy_staging_pair(p);
                true
            }
            None => false,
        }
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

    /// Whether a `BatchFull` or `DescriptorsExhausted` flush submits without
    /// waiting: `ACK_ASYNC_FLUSH` at open (on by default) with the timestamps
    /// instrument off. Every other flush waits for its batch either way; this is
    /// `completes`'s rule, stated for a caller outside the crate.
    pub fn flush_is_asynchronous(&self) -> bool {
        self.async_flush.get() && !self.ts_enabled.get()
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
            tags: Vec::new(),
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
    /// The staging pairs of a batch whose wait FAILED, or that was lost:
    /// destroyed, or leaked and counted on a poisoned context whose device
    /// state is unknown. A batch whose wait succeeded recycles them instead
    /// (`recycle_staging`).
    fn release_staging(&self, staging: Vec<StagingPair>) {
        if staging.is_empty() {
            return;
        }
        let objects = staging.len() as u64 * 2;
        self.release_or_leak(objects, || {
            for pair in staging {
                self.destroy_staging_pair(pair);
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
        // The same rule per dispatch tag: every item this batch tagged is
        // unresolved until its interval is read, so a lost batch stays visible
        // in its tags' rows too.
        self.tags_pending(&rec.tags);

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
        // R2 and the stream hooks: see `one_shot`'s identical block. With
        // nothing armed (every flush before R2, and any after it with nothing
        // armed) the submit info is exactly the line above.
        let (waits, signals) = self.take_submission_sync();
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = waits.iter().map(|w| w.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signal_semaphores: Vec<vk::Semaphore> = signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<u64> = signals.iter().map(|s| s.1).collect();
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let mut submit_info = vk::SubmitInfo::default().command_buffers(&cbs);
        if !waits.is_empty() {
            submit_info = submit_info
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }
        if !signals.is_empty() {
            submit_info = submit_info.signal_semaphores(&signal_semaphores);
        }
        if !waits.is_empty() || !signals.is_empty() {
            submit_info = submit_info.push_next(&mut timeline_info);
        }
        let submit = [submit_info];
        let timing_started = std::time::Instant::now();
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, &submit, fence) } {
            self.release_signals(&signals);
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
            tags: rec.tags,
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
            tags,
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
        if stamps > 1 && !ts_overflow && self.accumulate_timestamps(stamps, host_elapsed_ns, &tags)
        {
            self.ts_unresolved.set(self.ts_unresolved.get() - 1);
        }
        self.release_retired(retired, retired_kernels);
        for pair in retired_staging {
            self.recycle_staging(pair);
        }
        self.release_completed_submission(fence, free_cb);
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
    ///
    /// PER TAG (2026-10-03): interval `i` is charged to `tags[i]`, the tag its
    /// item was recorded under, and leaves that tag's unresolved count. The
    /// per-tag sums therefore partition `device_ns` exactly. Tags of a length
    /// other than the intervals' (which the recorder never produces) are left
    /// unresolved rather than charged by position to the wrong items.
    fn accumulate_timestamps(&self, stamps: u32, host_elapsed_ns: f64, tags: &[u32]) -> bool {
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
        let Ok(intervals) = dispatch_intervals(
            &ticks,
            self.timestamp_valid_bits,
            f64::from(self.report.timestamp_period_ns),
            host_elapsed_ns,
        ) else {
            return false;
        };
        let (sum_ns, min_ns, max_ns) = summarize_intervals(&intervals);
        if tags.len() == intervals.len() {
            self.tags_resolved(tags, &intervals);
        }
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

    /// Charge the completion interval of every item recorded from now on to
    /// `tag`, until the next call. 0 means untagged, and a context opens with
    /// it. The caller names its tags (`ack_torch.DeviceTimeByOp` gives each
    /// aten operation one); the kit keeps only the number.
    ///
    /// NOT REFUSED WITH A BATCH OPEN, unlike `set_dispatch_timestamps`: a tag
    /// is read per item when the item is recorded, so one batch can hold items
    /// of several tags. With the timestamps off nothing is recorded per tag.
    pub fn set_dispatch_tag(&self, tag: u32) {
        self.disp_tag.set(tag);
    }

    /// The tag `set_dispatch_tag` set last; 0 when none.
    pub fn dispatch_tag(&self) -> u32 {
        self.disp_tag.get()
    }

    /// The timed items' completion intervals, summed per dispatch tag, in tag
    /// order. Empty until a batch recorded with the timestamps on is flushed.
    ///
    /// THE ROWS PARTITION THE INSTRUMENT. Each interval read back is charged
    /// to exactly one tag, so the rows' `device_ns` sum to
    /// `dispatch_stats().device_ns` and their `items` to its `timed_items`.
    ///
    /// WHAT AN INTERVAL IS: the one `device_ns` sums, the ordered completion
    /// interval between an item's bottom-of-pipe stamp and the stamp before it
    /// (the batch origin for a batch's first item). Under R1's barrier between
    /// adjacent items, that is the item's barrier drain, launch, arithmetic and
    /// memory traffic. A transfer recorded between two items (an upload) falls
    /// in the next item's interval. It is not an occupancy figure. And with the
    /// timestamps on every flush completes (`completes`), so the batches under
    /// the instrument are not the batches without it.
    ///
    /// `unresolved_items` counts a tag's items whose interval was not read:
    /// their batch overflowed the query pool, failed, or had an ambiguous
    /// wrap. Nonzero means that tag's `device_ns` covers less than its work.
    pub fn dispatch_tag_times(&self) -> Vec<DispatchTagTime> {
        self.tag_times.borrow().values().copied().collect()
    }

    /// Every item a flushed batch tagged starts unresolved; `tags_resolved`
    /// moves it to its tag's totals once its interval is read.
    fn tags_pending(&self, tags: &[u32]) {
        if !tags.is_empty() {
            tag_rows_pending(&mut self.tag_times.borrow_mut(), tags);
        }
    }

    /// Interval `i`, read for an item recorded under `tags[i]`: charged to
    /// that tag, and no longer unresolved.
    fn tags_resolved(&self, tags: &[u32], intervals: &[u64]) {
        tag_rows_resolved(&mut self.tag_times.borrow_mut(), tags, intervals);
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
            for pair in rec.retired_staging {
                self.recycle_staging(pair);
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
        if self.fence_pool_enabled.get() {
            let pooled = self.fence_pool.borrow_mut().pop();
            if let Some(f) = pooled {
                // Already a live, counted object and unsignalled (reset when it
                // was pooled). The fault injector still fires at this step, so a
                // pool hit cannot leave an armed `CreateStep::Fence` fault
                // unconsumed and turn a cleanup-path test into a test of nothing.
                if let Some(r) = self.take_create_fault(CreateStep::Fence) {
                    self.fence_pool.borrow_mut().push(f);
                    return Err(r);
                }
                return Ok(f);
            }
        }
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

    /// R1/R2: the transfer queue's own command buffer, from
    /// `transfer_command_pool` rather than `command_pool`. Same counting and
    /// same fault (`CreateStep::CommandBuffer` does not distinguish which
    /// pool a command buffer came from -- it is one creation either way).
    /// Panics if called with no transfer command pool; every caller checks
    /// `self.transfer_queue`/`self.timeline` first and takes R2's fallback
    /// instead when either is absent.
    fn vk_allocate_transfer_command_buffer(
        &self,
    ) -> std::result::Result<vk::CommandBuffer, vk::Result> {
        let pool = self
            .transfer_command_pool
            .expect("vk_allocate_transfer_command_buffer: no transfer command pool");
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { self.device.allocate_command_buffers(&alloc) }?[0];
        self.track(1);
        if let Some(r) = self.take_create_fault(CreateStep::CommandBuffer) {
            self.vk_free_transfer_command_buffer(cb);
            return Err(r);
        }
        Ok(cb)
    }

    fn vk_free_transfer_command_buffer(&self, cb: vk::CommandBuffer) {
        let pool = self
            .transfer_command_pool
            .expect("vk_free_transfer_command_buffer: no transfer command pool");
        unsafe { self.device.free_command_buffers(pool, &[cb]) };
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

    /// The two families a `Context::buffer` or `Context::staging` allocation
    /// shares across (R1), when this device has a transfer queue; `None`
    /// when it does not, which callers turn into `BufferSharing::Exclusive`.
    fn transfer_shared_families(&self) -> Option<[u32; 2]> {
        self.transfer_queue_family.map(|tf| [self.queue_family, tf])
    }

    fn raw_buffer(
        &self,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
        sharing: BufferSharing,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let mut info = vk::BufferCreateInfo::default().size(bytes).usage(usage);
        info = match sharing {
            BufferSharing::Exclusive => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
            BufferSharing::Concurrent(families) => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
        };
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
        //
        // WITH THE POOL ON, THE PAIR IS ALLOCATED AT THE REQUEST'S SIZE CLASS
        // (`pool_class`), and the pool keys on it. Measured 2026-09-29 on the
        // realistic stage-E step: the MoE bmm and fp16 mul requests take
        // per-expert sizes that change every step, so an exact-size key missed
        // 1.2-1.6k times a step under either eviction policy. A class holds
        // every request within 12.5% below it, so those requests share pairs;
        // the price is at most that much device memory per buffer. Below
        // `POOL_CLASS_FLOOR` the key stays exact, where the rounding would buy
        // nothing. The refusal above is on the request, so a request the device
        // can hold is never refused because its class is larger
        // (`pool_class` clamps at the limit).
        let allocated = if self.pool_enabled.get() {
            pool_class(allocated, self.max_storage_buffer_bytes)
        } else {
            allocated
        };
        // A hit's Buffer records the PAIR's allocated size, which is larger than
        // the request's class when `pool_pick` took a larger pair (2026-10-01).
        let (buffer, memory, allocated) = match self.take_pooled(allocated) {
            Some(p) => {
                self.pool_hits.set(self.pool_hits.get() + 1);
                (p.buffer, p.memory, p.allocated)
            }
            None => {
                self.pool_misses.set(self.pool_misses.get() + 1);
                // GIVE MEMORY BACK BEFORE THE BUDGET, NOT ONLY AFTER A FAILURE. On Windows (WDDM) an allocation past
                // the device-local budget does not fail: the OS places it in system memory, and every dispatch that
                // touches it then crosses the bus. Measured 2026-10-21 (RX 9070 XT, a 26M-parameter training step at
                // microbatch 32): torch's live peak 12.3 GB plus a pool of up to half the heap ran 13x slower than
                // microbatch 16, with no allocation ever failing, so the retry below never ran. When the OS reports
                // less room than this request plus `POOL_BUDGET_MARGIN`, the largest pooled pairs go first.
                if let Some(room) = self.device_local_room() {
                    let need = allocated.saturating_add(POOL_BUDGET_MARGIN);
                    if room < need && self.pooled_buffer_objects() > 0 {
                        self.pool_reclaims.set(self.pool_reclaims.get() + 1);
                        self.evict_largest(need - room);
                    }
                }
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST;
                let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                // R1: shared across both queue families so a later
                // `copy_from_staging_async` can hand this buffer to the
                // transfer queue as `dst` while the compute queue also binds
                // it; `Exclusive` on a device with no transfer queue, which
                // is the pre-R1 behaviour exactly.
                let shared = self.transfer_shared_families();
                let sharing = match &shared {
                    Some(f) => BufferSharing::Concurrent(f),
                    None => BufferSharing::Exclusive,
                };
                let (b, m) = match self.raw_buffer(allocated, usage, flags, sharing) {
                    Ok(pair) => pair,
                    Err(e) => {
                        // THE POOL MUST NEVER BE THE REASON AN ALLOCATION
                        // FAILED. It is holding device memory that nothing
                        // owns, so an allocation failure with a non-empty pool
                        // is the one case where the right answer is certain:
                        // give memory back and try again. Without this the
                        // ceiling is a SAFETY mechanism and has to be set low;
                        // with it the ceiling is a pure tuning knob, and that
                        // is what lets the default be raised.
                        //
                        // GIVE BACK WHAT THE REQUEST NEEDS, NOT EVERYTHING.
                        // This drained the whole pool at the first failure.
                        // Measured 2026-09-29 on the realistic stage-E step
                        // (window 5 attribution, then window 6's timed steps):
                        // 2 such drains in one attributed run, each destroying
                        // ~8.5 GB of pooled pairs, and the steps after them
                        // allocated from nothing -- the old pool's step times
                        // were 8.9 / 14.2 / 8.6 / 7.9 / 7.1 s, spiking where the
                        // pool refilled. Now the largest held pairs go first,
                        // at least the request's size, doubling after each
                        // failed retry, so a failure costs O(log) retries and
                        // keeps what the next requests will take. A failure
                        // with the pool empty is a real out-of-memory and is
                        // returned.
                        if self.pooled_buffer_objects() == 0 {
                            return Err(e);
                        }
                        self.pool_reclaims.set(self.pool_reclaims.get() + 1);
                        let mut want = allocated;
                        loop {
                            let freed = self.evict_largest(want);
                            match self.raw_buffer(allocated, usage, flags, sharing) {
                                Ok(pair) => break pair,
                                Err(again) if freed == 0 => return Err(again),
                                Err(_) => want = want.saturating_mul(2),
                            }
                        }
                    }
                };
                (b, m, allocated)
            }
        };
        let id = self.next_buffer.get();
        self.next_buffer.set(id + 1);
        self.live_buffers.borrow_mut().insert(id);
        Ok(Buffer {
            buffer,
            memory,
            bytes,
            allocated,
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
    ///
    /// R2: both this and `flush` consult `take_pending_transfer_wait` at
    /// their own `queue_submit`, so "the next submission's wait list"
    /// (`await_transfer`'s doc comment) means the next call to either.
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
        // R2: when `await_transfer` armed a wait, THIS submission (the next
        // one on the compute queue, whichever call reaches `queue_submit`
        // first) carries it and consumes it; with nothing armed (every call
        // before R2, and every call after it that never used
        // `copy_from_staging_async`/`await_transfer`) `wait` is `None` and
        // the submit info below is byte-identical to before R2.
        //
        // The stream hook (`arm_submission_signal`) adds signals the same
        // way; with none armed it adds nothing either.
        let (waits, signals) = self.take_submission_sync();
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = waits.iter().map(|w| w.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signal_semaphores: Vec<vk::Semaphore> = signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<u64> = signals.iter().map(|s| s.1).collect();
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let mut submit_info = vk::SubmitInfo::default().command_buffers(&cbs);
        if !waits.is_empty() {
            submit_info = submit_info
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }
        if !signals.is_empty() {
            submit_info = submit_info.signal_semaphores(&signal_semaphores);
        }
        if !waits.is_empty() || !signals.is_empty() {
            submit_info = submit_info.push_next(&mut timeline_info);
        }
        let submit = [submit_info];
        if let Err(e) = unsafe { self.device.queue_submit(self.queue, &submit, fence) } {
            self.release_signals(&signals);
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
        self.release_completed_submission(fence, free_cb);
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
        let clock = transfer_clock_on().then(std::time::Instant::now);
        if self.deferred_upload() {
            let recorded = self.upload_recorded(dst, data);
            if let (Some(t0), Ok(())) = (clock, &recorded) {
                // a recorded upload: no flush, so its whole time is the rest
                transfer_clock_add(3, 0, clock_ns(t0, std::time::Instant::now()));
                TRANSFER_CLOCK[6].fetch_add(1, Ordering::Relaxed);
            }
            return recorded;
        }
        // after every named refusal, before any object of this call exists
        self.flush(batch::FlushReason::Upload)?;
        let flushed = clock.map(|_| std::time::Instant::now());
        // this call's own staging pair (pooled or fresh, mapped either way;
        // the transfer queue never touches it: R1 leaves `upload` unchanged)
        let pair = self.take_staging(TransferKind::Upload, bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), pair.ptr as *mut u8, data.len());
        }
        let staging = pair.buffer;
        let submitted = self.one_shot(|cb| unsafe {
            let region = [vk::BufferCopy::default().size(bytes)];
            self.device
                .cmd_copy_buffer(cb, staging, dst.buffer, &region);
        });
        match submitted {
            // the copy completed: nothing names the pair any more
            Ok(()) => self.recycle_staging(pair),
            // released when the device is quiesced or lost, left allocated and
            // counted when the context is poisoned with the device state unknown
            Err(_) => self.release_or_leak(2, || self.destroy_staging_pair(pair)),
        }
        if let (Some(t0), Some(t1), Ok(())) = (clock, flushed, &submitted) {
            transfer_clock_add(3, clock_ns(t0, t1), clock_ns(t1, std::time::Instant::now()));
        }
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
        // the same staging as `upload`'s own, pooled or fresh
        let pair = self.take_staging(TransferKind::Upload, bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), pair.ptr as *mut u8, data.len());
        }
        let staging = pair.buffer;
        if let Err(e) = self.open_batch() {
            // nothing was recorded: nothing names the pair
            self.recycle_staging(pair);
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
            rec.retired_staging.push(pair);
            rec.plan.due()
        };
        if let Some(reason) = due {
            self.flush(reason)?;
        }
        Ok(())
    }

    /// A persistent, mapped staging area of `bytes`, allocated with `kind`'s
    /// memory flags and mapped once for the CALLER's lifetime, rather than
    /// re-created on every call the way `upload`'s own staging buffer is. For
    /// a caller that will land bytes in device-visible memory more than
    /// once -- a memcpy, an unbuffered file read straight into the mapping
    /// (`FILE_FLAG_NO_BUFFERING`), or, for `DeviceLocalHostVisible`, a write
    /// straight into VRAM over the BAR -- this pays the allocation and the
    /// map once and lets the caller reuse both. See
    /// `docs/audits/2026-09-14-split-k-gemv-wave32-and-delivery.md`, Gate 1:
    /// `upload` delivers 5 GB/s on a 37-54 GB/s link because it allocates a
    /// fresh `HOST_VISIBLE | HOST_COHERENT` buffer, memcpys into that
    /// write-combined memory and fences, per call.
    ///
    /// NOT POOLED, NOT RECORDED, AND NO CHANGE TO `upload`: this is a second,
    /// explicit path next to it for a caller that wants to own the staging
    /// memory's lifetime itself, not a replacement for the eager one-shot
    /// path `upload` and `upload_recorded` still take.
    ///
    /// `check_live`, then `raw_buffer(bytes, STORAGE_BUFFER | TRANSFER_SRC,
    /// flags)` with flags chosen by `kind`. A memory type that does not carry
    /// them is refused by name (`AckError::Unsupported`, naming `kind`)
    /// before any object exists: `raw_buffer` itself destroys the `VkBuffer`
    /// it tried before returning that error (ACK-L0-05), so this only renames
    /// it. A failure to map the pair once it exists destroys it the same way
    /// before returning (ACK-L0-05, as `upload`'s own staging creation does).
    /// Every kind's memory is `HOST_COHERENT`, so the mapping needs no flush
    /// or invalidate on either side of a copy.
    ///
    /// R3: `STORAGE_BUFFER` is new alongside the `TRANSFER_SRC` this always
    /// had, so the returned area can be bound as a kernel operand
    /// (`Staging::as_operand`) with no intervening `copy_from_staging`. If
    /// adding `STORAGE_BUFFER` costs a `kind` its only memory type on this
    /// device, `raw_buffer`'s `memory_type` lookup fails and the `kind`-named
    /// refusal below fires exactly as it already does for any other
    /// unsupported kind -- nothing here silently drops back to a kind the
    /// caller did not ask for.
    ///
    /// R1: shared across both queue families exactly like `Context::buffer`'s
    /// device-local buffers, and for the same reason -- a `Staging` area can
    /// be `copy_from_staging_async`'s `src` on the transfer queue while the
    /// compute queue binds it as an operand (R3) or reads the destination
    /// buffer the copy fills. `Exclusive` on a device with no transfer queue.
    pub fn staging(&self, bytes: u64, kind: StagingKind) -> Result<Staging> {
        self.staging_with(bytes, kind, false)
    }

    /// A persistent, mapped READBACK area: `staging(bytes, HostCached)`
    /// whose buffer is also a transfer destination, so `record_readback` can
    /// copy device bytes into it inside a batch. Cached, because the host
    /// READS this memory (see `readback_memory_preferences`). Destroyed by
    /// `destroy_staging` like any other `Staging`.
    pub fn readback_area(&self, bytes: u64) -> Result<Staging> {
        self.staging_with(bytes, StagingKind::HostCached, true)
    }

    fn staging_with(&self, bytes: u64, kind: StagingKind, readback: bool) -> Result<Staging> {
        self.check_live()?;
        let flags = match kind {
            StagingKind::HostCoherent => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            StagingKind::HostCached => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED
            }
            StagingKind::DeviceLocalHostVisible => {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        };
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        if readback {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let shared = self.transfer_shared_families();
        let sharing = match &shared {
            Some(f) => BufferSharing::Concurrent(f),
            None => BufferSharing::Exclusive,
        };
        let (buffer, memory) = match self.raw_buffer(bytes, usage, flags, sharing) {
            Ok(pair) => pair,
            Err(AckError::Unsupported(msg)) => {
                return Err(AckError::Unsupported(format!(
                    "staging kind {kind:?}: {msg}"
                )));
            }
            Err(e) => return Err(e),
        };
        let release = || {
            self.vk_destroy_buffer(buffer);
            self.vk_free_memory(memory);
        };
        let ptr = unsafe {
            match self
                .device
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
            {
                Ok(p) => p.cast::<u8>(),
                Err(e) => {
                    // nothing was submitted: the pair is safe to destroy (ACK-L0-05)
                    release();
                    return Err(e.into());
                }
            }
        };
        Ok(Staging {
            buffer,
            memory,
            ptr,
            bytes,
            kind,
            readback,
        })
    }

    /// Copy `bytes` from `src` (a `Staging` area, at `src_offset`) into `dst`
    /// (at `dst_offset`). EAGER, like `upload`: `check_live`, `check_buffer`
    /// on `dst`, then a zero-length or an out-of-range copy on either side is
    /// refused by name before any Vulkan call (review finding ACK-L0-01).
    /// Otherwise this flushes the recorded batch first (`FlushReason::Upload`
    /// -- the same reason `upload` flushes for, since this is another way of
    /// landing bytes in a device buffer) and then records and waits for one
    /// `cmd_copy_buffer`, exactly as `upload`'s own copy does.
    ///
    /// `src` is the caller's: this neither destroys nor unmaps it, and the
    /// same `Staging` can be copied from again, at the same or a different
    /// offset, as soon as this returns.
    pub fn copy_from_staging(
        &self,
        src: &Staging,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        self.check_live()?;
        self.check_buffer(dst)?;
        let in_range = match (src_offset.checked_add(bytes), dst_offset.checked_add(bytes)) {
            (Some(src_end), Some(dst_end)) => src_end <= src.bytes && dst_end <= dst.bytes,
            _ => false,
        };
        if bytes == 0 || !in_range {
            return Err(AckError::Unsupported(format!(
                "copy_from_staging of {bytes} bytes: staging offset {src_offset} of {} bytes, buffer offset {dst_offset} of {} bytes",
                src.bytes, dst.bytes
            )));
        }
        self.flush(batch::FlushReason::Upload)?;
        self.one_shot(|cb| unsafe {
            let region = [vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(bytes)];
            self.device
                .cmd_copy_buffer(cb, src.buffer, dst.buffer, &region);
        })
    }

    /// `copy_from_staging` for many regions in ONE submission: each `(src, src_offset, dst, dst_offset, bytes)`
    /// is checked as `copy_from_staging` checks its one, the batch is flushed once, and one command buffer records
    /// every copy and is waited for once. Regions run in the order given; an empty list does nothing.
    pub fn copy_from_stagings(&self, regions: &[(&Staging, u64, &Buffer, u64, u64)]) -> Result<()> {
        self.check_live()?;
        if regions.is_empty() {
            return Ok(());
        }
        for (i, &(src, src_offset, dst, dst_offset, bytes)) in regions.iter().enumerate() {
            self.check_buffer(dst)?;
            let in_range = match (src_offset.checked_add(bytes), dst_offset.checked_add(bytes)) {
                (Some(src_end), Some(dst_end)) => src_end <= src.bytes && dst_end <= dst.bytes,
                _ => false,
            };
            if bytes == 0 || !in_range {
                return Err(AckError::Unsupported(format!(
                    "copy_from_stagings region {i} of {bytes} bytes: staging offset {src_offset} of {} bytes, buffer offset {dst_offset} of {} bytes",
                    src.bytes, dst.bytes
                )));
            }
        }
        self.flush(batch::FlushReason::Upload)?;
        self.one_shot(|cb| unsafe {
            for &(src, src_offset, dst, dst_offset, bytes) in regions {
                let region = [vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(bytes)];
                self.device
                    .cmd_copy_buffer(cb, src.buffer, dst.buffer, &region);
            }
        })
    }

    /// RECORDED device-to-device copies, in the open batch: each region's
    /// `bytes` from `src` at `src_offset` into `dst` at `dst_offset`. Nothing
    /// is flushed or waited for, and the bytes are copied as they are (a
    /// transfer, not a shader: no float instruction sees them).
    ///
    /// ORDERING. The copies run where they are recorded. One barrier precedes
    /// the group when an item since the last barrier wrote a source or
    /// touched a destination, or when another recorded transfer is pending;
    /// the next dispatch takes a barrier after them (`upload_recorded`'s
    /// rule). Refused by name, before anything is recorded: an empty or
    /// out-of-range region, a region whose source and destination are one
    /// buffer, destinations that overlap one another, and a buffer that is a
    /// source of one region and a destination of another, so the group needs
    /// no barrier inside it.
    pub fn record_buffer_copies(&self, regions: &[BufferCopy<'_>]) -> Result<()> {
        self.check_live()?;
        for (i, r) in regions.iter().enumerate() {
            self.check_buffer(r.src)?;
            self.check_buffer(r.dst)?;
            let in_range = match (
                r.src_offset.checked_add(r.bytes),
                r.dst_offset.checked_add(r.bytes),
            ) {
                (Some(src_end), Some(dst_end)) => src_end <= r.src.bytes && dst_end <= r.dst.bytes,
                _ => false,
            };
            if r.bytes == 0 || !in_range || r.src.id == r.dst.id {
                return Err(AckError::Unsupported(format!(
                    "record_buffer_copies region {i} of {} bytes: source offset {} of {} bytes, destination offset {} of {} bytes, one buffer: {}",
                    r.bytes,
                    r.src_offset,
                    r.src.bytes,
                    r.dst_offset,
                    r.dst.bytes,
                    r.src.id == r.dst.id
                )));
            }
            if let Some(j) = regions[..i].iter().position(|o| {
                (o.dst.id == r.dst.id
                    && o.dst_offset < r.dst_offset + r.bytes
                    && r.dst_offset < o.dst_offset + o.bytes)
                    || o.dst.id == r.src.id
                    || o.src.id == r.dst.id
            }) {
                return Err(AckError::Unsupported(format!(
                    "record_buffer_copies: regions {j} and {i} overlap, or one reads what the other writes"
                )));
            }
        }
        if regions.is_empty() {
            return Ok(());
        }
        self.open_batch()?;
        let mut slot = self.recorder.borrow_mut();
        let rec = slot
            .as_mut()
            .expect("open_batch installed the recorder or returned an error");
        let reads: Vec<u64> = regions.iter().map(|r| r.src.id).collect();
        let writes: Vec<u64> = regions.iter().map(|r| r.dst.id).collect();
        if (rec.plan.record_transfer_group(&reads, &writes) || rec.pending_transfer)
            && self.barriers_enabled()
        {
            self.record_barrier(rec.cb);
        }
        for r in regions {
            let region = [vk::BufferCopy::default()
                .src_offset(r.src_offset)
                .dst_offset(r.dst_offset)
                .size(r.bytes)];
            unsafe {
                self.device
                    .cmd_copy_buffer(rec.cb, r.src.buffer, r.dst.buffer, &region)
            };
        }
        rec.pending_transfer = true;
        Ok(())
    }

    /// RECORDED copies out of a persistent `Staging` area into device
    /// buffers, in the open batch. Nothing is flushed or waited for, and no
    /// staging of the kit's own is taken: this is `upload_recorded` for a
    /// caller that owns its staging memory, for several destinations at once.
    ///
    /// ORDERING. The copies run where they are recorded: after every item
    /// recorded before them, before every item recorded after them. One
    /// barrier precedes the group when an item since the last barrier
    /// touched a destination, or when another recorded transfer is pending.
    /// The next dispatch takes a barrier after them, whatever its place in
    /// the batch (`upload_recorded`'s rule). Destinations that overlap one
    /// another are refused by name, so the group needs no barrier inside it.
    ///
    /// THE CALLER OWNS THE BYTES' LIFETIME. The device reads the area when
    /// the batch executes, not when this returns, so a region must not be
    /// rewritten until the batch carrying its copy has completed. Under the
    /// timestamp instrument the copies' time falls in the next item's
    /// interval.
    pub fn record_copies_from_staging(
        &self,
        src: &Staging,
        regions: &[StagingCopy<'_>],
    ) -> Result<()> {
        self.check_live()?;
        for (i, r) in regions.iter().enumerate() {
            self.check_buffer(r.dst)?;
            let in_range = match (
                r.staging_offset.checked_add(r.bytes),
                r.dst_offset.checked_add(r.bytes),
            ) {
                (Some(src_end), Some(dst_end)) => src_end <= src.bytes && dst_end <= r.dst.bytes,
                _ => false,
            };
            if r.bytes == 0 || !in_range {
                return Err(AckError::Unsupported(format!(
                    "record_copies_from_staging region {i} of {} bytes: staging offset {} of {} bytes, buffer offset {} of {} bytes",
                    r.bytes, r.staging_offset, src.bytes, r.dst_offset, r.dst.bytes
                )));
            }
            if let Some(j) = regions[..i].iter().position(|o| {
                o.dst.id == r.dst.id
                    && o.dst_offset < r.dst_offset + r.bytes
                    && r.dst_offset < o.dst_offset + o.bytes
            }) {
                return Err(AckError::Unsupported(format!(
                    "record_copies_from_staging: regions {j} and {i} overlap in their destination"
                )));
            }
        }
        if regions.is_empty() {
            return Ok(());
        }
        self.open_batch()?;
        let mut slot = self.recorder.borrow_mut();
        let rec = slot
            .as_mut()
            .expect("open_batch installed the recorder or returned an error");
        let ids: Vec<u64> = regions.iter().map(|r| r.dst.id).collect();
        if (rec.plan.record_transfer_writes(&ids) || rec.pending_transfer)
            && self.barriers_enabled()
        {
            self.record_barrier(rec.cb);
        }
        for r in regions {
            let region = [vk::BufferCopy::default()
                .src_offset(r.staging_offset)
                .dst_offset(r.dst_offset)
                .size(r.bytes)];
            unsafe {
                self.device
                    .cmd_copy_buffer(rec.cb, src.buffer, r.dst.buffer, &region)
            };
        }
        rec.pending_transfer = true;
        Ok(())
    }

    /// A RECORDED copy of `bytes` from a device buffer into a readback area
    /// (`readback_area`), in the open batch, followed by a transfer-to-host
    /// memory barrier. Nothing is flushed or waited for. The bytes are in
    /// `dst.as_slice()` once the batch carrying the copy has completed: a
    /// fence the kit waited on, or a timeline value that batch signalled
    /// (`arm_submission_signal`) and the host has waited past.
    ///
    /// ORDERING. One barrier precedes the copy when an item since the last
    /// barrier wrote `src` (read after write), or when another recorded
    /// transfer is pending. The read is tracked, so a later item that writes
    /// `src` waits for it. Refused by name: an area that is not a readback
    /// area, and an empty or out-of-range copy.
    pub fn record_readback(
        &self,
        src: &Buffer,
        src_offset: u64,
        dst: &Staging,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        self.check_live()?;
        self.check_buffer(src)?;
        if !dst.readback {
            return Err(AckError::Unsupported(
                "record_readback into a staging area that is not a transfer destination: make it with Context::readback_area".into(),
            ));
        }
        let in_range = match (src_offset.checked_add(bytes), dst_offset.checked_add(bytes)) {
            (Some(src_end), Some(dst_end)) => src_end <= src.bytes && dst_end <= dst.bytes,
            _ => false,
        };
        if bytes == 0 || !in_range {
            return Err(AckError::Unsupported(format!(
                "record_readback of {bytes} bytes: buffer offset {src_offset} of {} bytes, area offset {dst_offset} of {} bytes",
                src.bytes, dst.bytes
            )));
        }
        self.open_batch()?;
        let mut slot = self.recorder.borrow_mut();
        let rec = slot
            .as_mut()
            .expect("open_batch installed the recorder or returned an error");
        if (rec.plan.record_transfer_read(src.id) || rec.pending_transfer)
            && self.barriers_enabled()
        {
            self.record_barrier(rec.cb);
        }
        let region = [vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(bytes)];
        // the device -> host domain operation a host read of device-written
        // memory needs, whatever a fence or semaphore wait adds on this driver
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        unsafe {
            self.device
                .cmd_copy_buffer(rec.cb, src.buffer, dst.buffer, &region);
            self.device.cmd_pipeline_barrier(
                rec.cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
        }
        rec.pending_transfer = true;
        Ok(())
    }

    /// The next value this context's timeline semaphore will be signalled to.
    fn next_timeline_value(&self) -> u64 {
        let v = self.timeline_value.get() + 1;
        self.timeline_value.set(v);
        v
    }

    /// Free every async transfer command buffer (R2) whose ticket value the
    /// timeline has already reached, so the small outstanding list does not
    /// grow across a pipeline's steady state. Called opportunistically from
    /// `copy_from_staging_async` (before recording the next one) and from
    /// `wait_transfer`; `Drop` reaps every remaining entry unconditionally,
    /// after a `device_wait_idle` that makes every one of them safe to free
    /// whatever its value.
    fn reap_transfer_command_buffers(&self) {
        let Some(timeline) = self.timeline else {
            return;
        };
        let current = unsafe { self.device.get_semaphore_counter_value(timeline) }.unwrap_or(0);
        let mut pending = self.pending_transfer_cbs.borrow_mut();
        let mut i = 0;
        while i < pending.len() {
            if pending[i].0 <= current {
                let (_, cb) = pending.remove(i);
                self.vk_free_transfer_command_buffer(cb);
            } else {
                i += 1;
            }
        }
    }

    /// R2: a copy recorded on the TRANSFER queue (R1), returning immediately
    /// without waiting for it. The returned ticket signals this context's
    /// timeline semaphore at a value only this copy sets; `await_transfer`
    /// and `wait_transfer` are how a caller orders later work after it.
    ///
    /// THE FALLBACK IS NOT A DIFFERENT CONTRACT, IT IS THE SAME ONE. If this
    /// device has no transfer queue (`report.transfer_queue`) or no timeline
    /// semaphore (`report.timeline_semaphores`), this does the ordinary
    /// synchronous `copy_from_staging` on the compute queue right here and
    /// returns a ticket that is already satisfied: `await_transfer` and
    /// `wait_transfer` both special-case a timeline-less context to a no-op,
    /// so a caller never branches on which arm it got -- it just calls
    /// `copy_from_staging_async`, then (eventually) `await_transfer` or
    /// `wait_transfer`, on every device.
    ///
    /// Refusals mirror `copy_from_staging`'s exactly (same bounds check, same
    /// message), checked before either arm touches Vulkan.
    pub fn copy_from_staging_async(
        &self,
        src: &Staging,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<TransferTicket> {
        self.check_live()?;
        self.check_buffer(dst)?;
        let in_range = match (src_offset.checked_add(bytes), dst_offset.checked_add(bytes)) {
            (Some(src_end), Some(dst_end)) => src_end <= src.bytes && dst_end <= dst.bytes,
            _ => false,
        };
        if bytes == 0 || !in_range {
            return Err(AckError::Unsupported(format!(
                "copy_from_staging_async of {bytes} bytes: staging offset {src_offset} of {} bytes, buffer offset {dst_offset} of {} bytes",
                src.bytes, dst.bytes
            )));
        }
        let (transfer_queue, timeline) = match (self.transfer_queue, self.timeline) {
            (Some(q), Some(t)) => (q, t),
            _ => {
                // R2's fallback: no transfer queue, or no timeline semaphore.
                // `value: 0` is never compared to anything -- `await_transfer`
                // and `wait_transfer` both return at once on a context with
                // no timeline, which this context, by construction, is.
                self.copy_from_staging(src, src_offset, dst, dst_offset, bytes)?;
                return Ok(TransferTicket { value: 0 });
            }
        };
        self.reap_transfer_command_buffers();
        let cb = self.vk_allocate_transfer_command_buffer()?;
        let free_cb = || self.vk_free_transfer_command_buffer(cb);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(e) = unsafe { self.device.begin_command_buffer(cb, &begin) } {
            free_cb();
            return Err(e.into());
        }
        unsafe {
            let region = [vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(bytes)];
            self.device
                .cmd_copy_buffer(cb, src.buffer, dst.buffer, &region);
        }
        if let Err(e) = unsafe { self.device.end_command_buffer(cb) } {
            free_cb();
            return Err(e.into());
        }
        let value = self.next_timeline_value();
        let cbs = [cb];
        let signal_semaphores = [timeline];
        let signal_values = [value];
        let mut timeline_signal =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let submit = [vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_signal)];
        // KNOWN SIMPLIFICATION (deviation, see the report): unlike `one_shot`
        // and `flush`, a failed submission here does not poison the context.
        // Nothing was recorded on the compute queue, and `queue_submit`'s own
        // guarantee about an unaffected queue on failure is the same limited
        // one `lose_batch` documents; extending ACK-L0-03's poison ladder to
        // a second queue is left for a follow-up rather than done in haste
        // here. The command buffer is freed and the error returned either way.
        if let Err(e) = unsafe {
            self.device
                .queue_submit(transfer_queue, &submit, vk::Fence::null())
        } {
            free_cb();
            return Err(e.into());
        }
        self.pending_transfer_cbs.borrow_mut().push((value, cb));
        Ok(TransferTicket { value })
    }

    /// R2: make every dispatch recorded after this call wait on `ticket`
    /// before it reads whatever the copy behind `ticket` wrote. Implemented
    /// as a wait on the timeline at `ticket`'s value in the NEXT submission's
    /// wait list (`one_shot` or `flush`, whichever the caller's next call
    /// reaches first) -- NOT as a host-side block. A single queue processes
    /// its submissions in the order they were made, so once one submission
    /// has waited past `ticket`'s value, every later submission to the same
    /// queue is already ordered after it and needs no wait of its own; that
    /// is why consuming the armed value once, at the very next submission,
    /// is enough (see `pending_transfer_wait`'s doc comment for the case of
    /// more than one outstanding `await_transfer`).
    ///
    /// A no-op on a context with no timeline semaphore: R2's fallback already
    /// ran the copy synchronously before handing back the ticket, so there is
    /// nothing left to order.
    pub fn await_transfer(&self, ticket: &TransferTicket) -> Result<()> {
        self.check_live()?;
        if self.timeline.is_none() {
            return Ok(());
        }
        let merged = match self.pending_transfer_wait.get() {
            Some(v) => v.max(ticket.value),
            None => ticket.value,
        };
        self.pending_transfer_wait.set(Some(merged));
        Ok(())
    }

    /// R2: host-side block until `ticket`'s copy has completed. For the
    /// probe's teardown and for tests, where a real wait (not just an
    /// ordering against the next dispatch) is what is wanted. Returns at once
    /// if `ticket` is already signalled -- `vkWaitSemaphores` does that on
    /// its own, and this adds nothing on top -- and at once, unconditionally,
    /// on a context with no timeline semaphore (R2's fallback: the copy
    /// already ran synchronously).
    pub fn wait_transfer(&self, ticket: &TransferTicket) -> Result<()> {
        self.check_live()?;
        let Some(timeline) = self.timeline else {
            return Ok(());
        };
        let semaphores = [timeline];
        let values = [ticket.value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }?;
        self.reap_transfer_command_buffers();
        Ok(())
    }

    /// Destroy a staging area this context created: unmap, then free and
    /// destroy its buffer and memory through the same release-or-leak
    /// accounting `destroy_buffer` uses (`release_or_leak(2, ..)`). Not
    /// pooled and never retired into a batch. A recorded copy
    /// (`record_copies_from_staging`, `record_readback`) names its area
    /// until that batch completes, so a caller that recorded one syncs
    /// before this; nothing else records a `Staging`.
    pub fn destroy_staging(&self, s: Staging) {
        unsafe { self.device.unmap_memory(s.memory) };
        self.release_or_leak(2, || {
            self.vk_destroy_buffer(s.buffer);
            self.vk_free_memory(s.memory);
        });
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
        let mut out = vec![0u8; src.bytes as usize];
        self.download_into(src, &mut out)?;
        Ok(out)
    }

    /// `download`, into the caller's own bytes; `dst` must be exactly `src`'s
    /// size. The C ABI's `ack_download` reads through this, so a readback
    /// makes one host copy (staging to `dst`) where it made three: a zeroed
    /// `Vec` of the whole size, the copy into it, and the ABI's copy out.
    ///
    /// ABOVE `READBACK_CHUNK_BYTES` THE COPY RUNS IN CHUNKS through ONE
    /// staging pair of that size, reused chunk after chunk, so no call stages
    /// more host memory than one chunk. It used to stage the whole buffer in a
    /// pair the pool never keeps (it only keeps pairs up to
    /// `STAGING_POOL_MAX_PAIR_BYTES`), and a fresh host-visible allocation
    /// that size was paid on every call. On NVIDIA that allocation pins pages.
    /// MEASURED on a Kaggle T4 (driver 580.159.04, 2026-09-27): `ack_download`
    /// read 0.46/0.69/0.80/0.68 GB/s at 1/4/16/64 MiB with the pool off, the
    /// crate's default. The torch adapter, pool on, read 1.8/2.5/0.43/0.44 GB/s,
    /// collapsing above the pooled size. CUDA on the same GPU read 10-13 GB/s
    /// into pinned memory and 2.8-4.2 GB/s into pageable memory, with the PCIe
    /// link at Gen3 throughout.
    ///
    /// A size refusal writes nothing. After a failure past it, the chunks
    /// already copied are in `dst` and the rest is unchanged: the caller
    /// treats all of `dst` as unspecified.
    pub fn download_into(&self, src: &Buffer, dst: &mut [u8]) -> Result<()> {
        self.check_live()?;
        self.check_buffer(src)?;
        if dst.len() as u64 != src.bytes {
            return Err(AckError::Unsupported(format!(
                "download of a {}-byte buffer into {} bytes",
                src.bytes,
                dst.len()
            )));
        }
        // after every named refusal, before any object of this call exists
        let clock = transfer_clock_on().then(std::time::Instant::now);
        self.flush(batch::FlushReason::Readback)?;
        let flushed = clock.map(|_| std::time::Instant::now());
        let total = src.bytes;
        let chunk = total.min(READBACK_CHUNK_BYTES);
        // this call's own readback staging (pooled or fresh, mapped either
        // way), in the memory type `readback_memory_preferences` picks: cached
        // is what makes the read fast. The transfer queue never touches it (R1
        // leaves `download` unchanged).
        let pair = self.take_staging(TransferKind::Readback, chunk)?;
        let staging = pair.buffer;
        let mut done = 0u64;
        while done < total {
            let size = (total - done).min(chunk);
            let submitted = self.one_shot(|cb| unsafe {
                let region = [vk::BufferCopy::default()
                    .src_offset(done)
                    .dst_offset(0)
                    .size(size)];
                self.device
                    .cmd_copy_buffer(cb, src.buffer, staging, &region);
            });
            if let Err(e) = submitted {
                // as in upload: destroy when quiesced or lost, leak and count when poisoned
                self.release_or_leak(2, || self.destroy_staging_pair(pair));
                return Err(e);
            }
            // the submission completed: nothing names the pair until the next
            // chunk's copy is recorded.
            // A non-coherent cached mapping must be invalidated before the
            // read, or the CPU may serve it from stale cache lines -- including
            // the previous chunk's, or a previous tenant's when the pair came
            // from the pool. WHOLE_SIZE is used rather than `size` because a
            // non-whole range must be aligned to nonCoherentAtomSize and
            // WHOLE_SIZE carries no such requirement.
            if !pair.coherent {
                let range = [vk::MappedMemoryRange::default()
                    .memory(pair.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                if let Err(e) = unsafe { self.device.invalidate_mapped_memory_ranges(&range) } {
                    self.destroy_staging_pair(pair);
                    return Err(e.into());
                }
            }
            // only the first `size` bytes of the pair: a pooled pair can be
            // larger than the chunk, and the rest is a previous tenant's
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pair.ptr as *const u8,
                    dst.as_mut_ptr().add(done as usize),
                    size as usize,
                );
            }
            done += size;
        }
        self.recycle_staging(pair);
        if let (Some(t0), Some(t1)) = (clock, flushed) {
            transfer_clock_add(0, clock_ns(t0, t1), clock_ns(t1, std::time::Instant::now()));
        }
        Ok(())
    }

    /// Build a compute pipeline from SPIR-V with `storage_buffers` storage-buffer
    /// bindings (set 0, bindings 0..n) and `push_bytes` of push constants.
    pub fn kernel(&self, spirv: &[u8], storage_buffers: u32, push_bytes: u32) -> Result<Kernel> {
        self.kernel_with_subgroup(spirv, storage_buffers, push_bytes, None)
    }

    /// Whether a compute pipeline may require `size` invocations per subgroup:
    /// subgroup size control is enabled, COMPUTE is among the
    /// requiredSubgroupSizeStages, and `size` is a power of two in
    /// [minSubgroupSize, maxSubgroupSize] (the valid-usage rules of
    /// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo).
    pub fn can_require_subgroup_size(&self, size: u32) -> bool {
        let (min, max) = self.subgroup_size_range;
        self.subgroup_size_control
            && self.required_subgroup_size_compute
            && size.is_power_of_two()
            && min <= size
            && size <= max
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
        if let Some(size) = required_subgroup_size {
            // Refused, not passed through: a size outside the device's range is
            // invalid usage that a driver need not reject. NVIDIA 580.173.02
            // on an L4 (range [32, 32]) accepted 64, and the 64-lane matmul
            // then returned wrong products with no error (2026-09-22).
            if !self.can_require_subgroup_size(size) {
                let (min, max) = self.subgroup_size_range;
                return Err(AckError::Unsupported(format!(
                    "kernel requires a subgroup size of {size}, which this device cannot guarantee \
                     (range [{min}, {max}], compute stage {})",
                    if self.required_subgroup_size_compute {
                        "admitted"
                    } else {
                        "not admitted"
                    }
                )));
            }
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
            bound_kinds: RefCell::new(Vec::new()),
            pending: RefCell::new(Vec::new()),
            reuse: Cell::new(None),
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
        kernel.reuse.set(None);
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
        // `bind` never writes `bound_kinds`; `check_dispatch` reads an empty
        // (or short) `bound_kinds` as "every slot is a `Buffer`", which is
        // `bind`'s only possible meaning and exactly its behaviour before R3.
        // Clearing it here (rather than leaving a longer, stale vector from
        // an earlier `bind_operand` call on the same kernel) keeps that
        // reading correct after a kernel is rebound with plain `bind`.
        kernel.bound_kinds.borrow_mut().clear();
        Ok(())
    }

    /// Bind buffers OR staging areas (R3) to the kernel's storage bindings in
    /// order: `Buffer::as_operand()` and `Staging::as_operand()` both produce
    /// the `BufferRef` this takes. Same validation as `bind`, generalised
    /// over `BufferRef` -- a tracked (`Buffer`-backed) operand is checked
    /// exactly as `bind` checks its `&Buffer` slice (`AckError::Foreign`,
    /// `AckError::Freed`); an untracked (`Staging`-backed) operand is
    /// trusted, the same way `copy_from_staging`'s `src: &Staging` already is.
    ///
    /// WHY A NEW METHOD RATHER THAN A NEW OVERLOAD OF `bind` (deviation from
    /// the spec's literal "`Context::bind` accepts a `Staging` wherever it
    /// accepts a `Buffer`", which the spec's own comment leaves room for --
    /// "or whatever the existing bind path takes"). `bind`'s signature,
    /// `&[&Buffer]`, is called at roughly twenty sites across this crate
    /// outside `src/lib.rs` and `src/ffi.rs` (kernels, probes, tests, the
    /// torch backend) that this job may not touch. Widening `bind` itself to
    /// take a trait-object or generic slice would compile at call sites that
    /// build the slice as a literal argument (`&[a, b]`, which coerces
    /// element-wise from the expected parameter type) but silently FAIL to
    /// compile at any call site that first names the slice's type some other
    /// way (`let xs = [a, b]; f(&xs)`, which infers `[&Buffer; N]` with no
    /// coercion left to apply) -- and there was no way to audit which of the
    /// twenty is which without editing files this job is scoped away from.
    /// `bind_operand` is additive and carries zero risk to any of them:
    /// `bind` keeps its exact signature and exact behaviour, unreached by
    /// anything below.
    ///
    /// A kernel bound with `bind_operand` and later re-bound with plain
    /// `bind` reverts to `bind`'s ordinary meaning (see `bind`'s own doc
    /// comment on `bound_kinds`).
    pub fn bind_operand(&self, kernel: &Kernel, operands: &[BufferRef]) -> Result<()> {
        self.check_live()?;
        self.check_kernel(kernel)?;
        for o in operands {
            if let Some((ctx_id, id)) = o.tracked {
                if ctx_id != self.id {
                    return Err(AckError::Foreign(format!(
                        "buffer {id} belongs to context {ctx_id}, not context {}",
                        self.id
                    )));
                }
                if !self.live_buffers.borrow().contains(&id) {
                    return Err(AckError::Freed(format!(
                        "buffer {id} of context {} was destroyed",
                        self.id
                    )));
                }
            }
        }
        if operands.len() as u32 != kernel.storage_buffers {
            return Err(AckError::Unsupported(format!(
                "kernel expects {} buffers, got {}",
                kernel.storage_buffers,
                operands.len()
            )));
        }
        kernel.reuse.set(None);
        *kernel.pending.borrow_mut() = operands
            .iter()
            .map(|o| {
                vk::DescriptorBufferInfo::default()
                    .buffer(o.buffer)
                    .offset(0)
                    .range(o.bytes)
            })
            .collect();
        // A `Staging` operand's slot gets the sentinel id 0 (`Context::buffer`
        // ids start at 1, so this can never collide with a real one) --
        // `bound_kinds` marks it so `check_dispatch` never looks 0 up in
        // `live_buffers`. If a caller later records this kernel's dispatch on
        // the batched `dispatch` path (rather than `dispatch_timed`), its
        // `Access` declaration for that slot is `Access::read(0)` /
        // `Access::write(0)`, matching the sentinel.
        *kernel.bound.borrow_mut() = operands
            .iter()
            .map(|o| o.tracked.map(|(_, id)| id).unwrap_or(0))
            .collect();
        *kernel.bound_kinds.borrow_mut() = operands.iter().map(|o| o.tracked.is_none()).collect();
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
    /// Turn descriptor-set reuse on or off for this context (tests; the
    /// default comes from `ACK_DESCRIPTOR_REUSE`).
    pub fn set_descriptor_reuse(&self, on: bool) {
        self.descriptor_reuse.set(on);
    }

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
            // R3: `bound_kinds[i]` is `true` exactly for a slot `bind_operand`
            // bound from a `Staging` (an untracked operand -- see
            // `BufferRef::tracked`'s doc comment for why it has no liveness
            // set of its own to check against). Every kernel bound only ever
            // through plain `bind` has an empty `bound_kinds`, so
            // `kinds.get(i)` is always `None` there and this loop is
            // byte-for-byte the pre-R3 one: `!live.contains(id)`, same order,
            // same message.
            let kinds = kernel.bound_kinds.borrow();
            if let Some((_, id)) = bound
                .iter()
                .enumerate()
                .find(|&(i, id)| !kinds.get(i).copied().unwrap_or(false) && !live.contains(id))
            {
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
        // a kernel whose bindings are unchanged since a set of THIS batch
        // generation was written reuses that set: the write would be identical
        let cached = match kernel.reuse.get() {
            Some((generation, set))
                if self.descriptor_reuse.get()
                    && self.arena_enabled()
                    && generation == self.batch_gen.get() =>
            {
                Some(set)
            }
            _ => None,
        };
        let set = match cached {
            Some(set) => {
                self.open_batch()?;
                set
            }
            None => {
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
                kernel.reuse.set(Some((self.batch_gen.get(), set)));
                set
            }
        };
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
                // the item's tag, stamped or not: an item past the capacity
                // is counted unresolved under its tag, not lost from it
                rec.tags.push(self.disp_tag.get());
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
        // the size the pair HAS (its class, if the pool was on when it was
        // made), never one re-derived from `bytes`: see `Buffer::allocated`
        let allocated = b.allocated;
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

/// A batch's completion intervals in nanoseconds, one per item: interval `i`
/// runs from stamp `i` to stamp `i + 1`. They are accepted together. The host
/// bound covers the whole submission; a possible extra counter wrap makes its
/// entire span unresolved, even when every modulo interval happens to look
/// small.
fn dispatch_intervals(
    ticks: &[u64],
    valid_bits: u32,
    period_ns: f64,
    host_elapsed_ns: f64,
) -> Result<Vec<u64>> {
    ticks
        .windows(2)
        .map(|pair| {
            resolve_timestamps(pair[0], pair[1], valid_bits, period_ns, host_elapsed_ns)
                .map(|ns| ns as u64)
        })
        .collect()
}

/// The sum, shortest and longest of a batch's intervals (`u64::MAX` and 0 for
/// none), as the instrument's totals fold them.
fn summarize_intervals(intervals: &[u64]) -> (u64, u64, u64) {
    intervals
        .iter()
        .fold((0u64, u64::MAX, 0u64), |(sum, min, max), &ns| {
            (sum.saturating_add(ns), min.min(ns), max.max(ns))
        })
}

/// `Context::tags_pending` on the rows alone: one unresolved item per tag
/// occurrence.
fn tag_rows_pending(rows: &mut BTreeMap<u32, DispatchTagTime>, tags: &[u32]) {
    for &tag in tags {
        rows.entry(tag)
            .or_insert(DispatchTagTime {
                tag,
                ..DispatchTagTime::default()
            })
            .unresolved_items += 1;
    }
}

/// `Context::tags_resolved` on the rows alone: interval `i` is charged to
/// `tags[i]` and leaves its unresolved count.
fn tag_rows_resolved(rows: &mut BTreeMap<u32, DispatchTagTime>, tags: &[u32], intervals: &[u64]) {
    for (&tag, &ns) in tags.iter().zip(intervals) {
        let row = rows.entry(tag).or_insert(DispatchTagTime {
            tag,
            ..DispatchTagTime::default()
        });
        row.unresolved_items = row.unresolved_items.saturating_sub(1);
        row.items += 1;
        row.device_ns = row.device_ns.saturating_add(ns);
        row.max_item_ns = row.max_item_ns.max(ns);
    }
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
        //
        // A signal armed for a submission that never came would leave a queue
        // that waits on it (the streaming pipeline's transfer queue) waiting
        // forever, and the idle below with it: signal those values from the
        // host first.
        let armed = std::mem::take(&mut *self.pending_signals.borrow_mut());
        self.release_signals(&armed);
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.abandon_batch();
        // the pool holds live Vulkan objects; drain them here, after the
        // abandon that may have added to it and before the device goes, or the
        // validation layer reports them at `vkDestroyDevice` under the
        // signature this crate assigns to a caller contract violation
        self.clear_pool();
        // and the fence pool's fences and the staging pool's pairs, for the
        // same reason (the abandon above releases the batch it may have been
        // holding, so this is the last)
        self.drain_fence_pool();
        self.drain_staging_pool();
        // the instrument's query pool, if it was ever created: after the idle
        // above, so no submitted batch can still be writing into it
        let qp = self.ts_pool.get();
        if qp != vk::QueryPool::null() {
            self.vk_destroy_query_pool(qp);
            self.ts_pool.set(vk::QueryPool::null());
        }
        // R1/R2: any async transfer command buffer this context has not yet
        // reaped. The `device_wait_idle` above is the same `VkDevice` and
        // therefore covers the transfer queue too, so every one of these has
        // completed and freeing it here is safe; `vk_free_transfer_command_buffer`
        // keeps `live_objects()` consistent (it is what tracked them as live).
        for (_, cb) in self.pending_transfer_cbs.borrow_mut().drain(..) {
            self.vk_free_transfer_command_buffer(cb);
        }
        unsafe {
            if let Some(sem) = self.timeline {
                self.device.destroy_semaphore(sem, None);
            }
            if let Some(pool) = self.transfer_command_pool {
                self.device.destroy_command_pool(pool, None);
            }
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

/// One dispatch tag's share of the timestamp instrument, read through
/// `Context::dispatch_tag_times` and the C entry `ack_dispatch_tag_times`
/// (2026-10-03).
///
/// WHY IT EXISTS. `DispatchStats::device_ns` says how long the device spent
/// on a step's items, and not on WHICH work. On 2026-10-02 a training step on
/// the RX 9070 XT was about 0.57 s of device time in a 0.62-0.66 s step, so
/// the next lever is kernel time, and choosing it needs that time split by
/// the operation that issued it. Splitting it before meant running every item
/// eagerly, one item per batch (`ACK_MAX_RECORDED_ITEMS=1`) and reading this
/// instrument around each operation, which is not the batching a real step
/// runs. A tag splits the batched span itself.
///
/// The rows partition the instrument: summed over tags, `items` is
/// `DispatchStats::timed_items` and `device_ns` is `DispatchStats::device_ns`.
/// An interval is what `device_ns` sums (see `Context::dispatch_tag_times`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DispatchTagTime {
    /// The caller's tag; 0 holds the items recorded with no tag set.
    pub tag: u32,
    /// Items recorded under this tag whose completion interval was read back.
    pub items: u64,
    /// Those intervals summed, in nanoseconds.
    pub device_ns: u64,
    /// The longest of them.
    pub max_item_ns: u64,
    /// Items recorded under this tag with the timestamps on whose interval was
    /// NOT read: their batch overflowed the query pool, failed, or had an
    /// ambiguous wrap. Nonzero means `device_ns` covers less than the tag's work.
    pub unresolved_items: u64,
}

/// The staging pool's counters (`Context::staging_pool_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StagingPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub returns: u64,
    pub declined: u64,
    /// Pooled pairs destroyed to make room for a SMALLER one (see
    /// `recycle_staging`).
    pub evicted: u64,
    pub held_pairs: i64,
    pub held_bytes: u64,
}

/// What the buffer pool has done, read through `Context::pool_stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// `Context::buffer` calls served without a Vulkan allocation.
    pub hits: u64,
    /// Of `hits`, those served by a pair of a LARGER class (`pool_pick`).
    pub hits_larger: u64,
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

/// R2: a copy `Context::copy_from_staging_async` recorded on the transfer
/// queue (or, on a device with no transfer queue or no timeline semaphore,
/// ran synchronously on the compute queue before returning). Opaque outside
/// this crate; the only public thing to do with one is name it to
/// `Context::await_transfer` or `Context::wait_transfer`, or read its raw
/// timeline value for the C ABI's ticket-as-`u64` convention (R4).
#[derive(Debug, Clone, Copy)]
pub struct TransferTicket {
    value: u64,
}

impl TransferTicket {
    /// This ticket's timeline value. On a context with no timeline semaphore
    /// this is `0` and means nothing on its own -- `await_transfer` and
    /// `wait_transfer` never compare it to the (nonexistent) timeline on such
    /// a context. The C ABI (R4) hands this back as the ticket's `u64` and
    /// reconstructs a `TransferTicket` from it the same way.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Rebuild a ticket from a raw value, for the C ABI's ticket-as-`u64`
    /// convention (R4): `ack_copy_from_staging_async` hands the caller
    /// `value()`, and `ack_await_transfer`/`ack_wait_transfer` rebuild a
    /// `TransferTicket` from it with this to call the safe API underneath.
    pub(crate) fn from_raw(value: u64) -> Self {
        Self { value }
    }
}

/// One kernel operand `Context::bind_operand` (R3) can bind: everything
/// `bind`'s own validation and descriptor write need, common to a `Buffer`
/// and to a `Staging` bound directly. Built by `Buffer::as_operand` or
/// `Staging::as_operand`; opaque otherwise.
///
/// `bind` ITSELF IS UNCHANGED -- see `Context::bind_operand`'s doc comment
/// for why binding a `Staging` is a new method rather than a new overload of
/// `bind`, and for the deviation that choice is from the spec's literal
/// wording.
#[derive(Clone, Copy)]
pub struct BufferRef {
    buffer: vk::Buffer,
    bytes: u64,
    /// `Some((ctx_id, id))` for a `Buffer` operand: checked exactly as `bind`
    /// checks its `&Buffer` slice (`AckError::Foreign`, `AckError::Freed`).
    /// `None` for a `Staging` operand -- `Staging` has never been tracked by
    /// the ownership contract (`copy_from_staging`'s `src: &Staging` is
    /// trusted the same way, with no Foreign or Freed check of its own), and
    /// `bind_operand` extends that existing precedent rather than inventing
    /// liveness tracking `Staging` has never had.
    tracked: Option<(u64, u64)>,
}

/// A device-local storage buffer; it knows which context made it.
pub struct Buffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub bytes: u64,
    /// The size `raw_buffer` created the `VkBuffer` with: `bytes` rounded up
    /// to a whole word and, with the pool on, to its size class
    /// (`pool_class`). Carried rather than re-derived from `bytes` at the free,
    /// because the class depends on whether the pool was on at the ALLOCATION,
    /// and the pool keys on this: a pair filed under a larger size than it has
    /// would serve a request its memory cannot hold.
    allocated: u64,
    ctx_id: u64,
    id: u64,
}

impl Buffer {
    /// The creating context's id.
    pub fn context_id(&self) -> u64 {
        self.ctx_id
    }

    /// The bytes of device memory behind this buffer: `bytes` rounded up to a
    /// whole word and, with the buffer pool on, to its size class. Descriptors
    /// still bind `bytes`, so no shader sees the difference.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated
    }

    /// This buffer's id within its context (the C ABI's sequence number is
    /// separate; this one is the safe runtime's).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// This buffer as a kernel operand, for `Context::bind_operand`'s mixed
    /// slots (R3) -- a plain `Buffer`-only kernel still uses `bind`, which is
    /// unchanged and does not need this.
    pub fn as_operand(&self) -> BufferRef {
        BufferRef {
            buffer: self.buffer,
            bytes: self.bytes,
            tracked: Some((self.ctx_id, self.id)),
        }
    }
}

/// Which memory a `Context::staging` area is allocated from. All three are
/// `HOST_COHERENT`; what differs is whether a CPU write, a CPU read, or a
/// device dispatch is the fast side, and whether the device exposes the type
/// at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingKind {
    /// `HOST_VISIBLE | HOST_COHERENT`: the type `upload` itself uses for its
    /// own, per-call staging buffer. Every Vulkan implementation has one.
    HostCoherent,
    /// `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`: a CPU read from this
    /// mapping goes through cache, the way `download`'s readback staging
    /// prefers (`readback_memory_preferences`). A CPU *write* does not need
    /// this, but a caller that will also read back what it just staged, or
    /// hand the mapping to an unbuffered file read, benefits from it.
    HostCached,
    /// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`: the resizable-BAR type
    /// this driver exposes on the device heap itself, so a CPU write (or an
    /// unbuffered file read) lands directly in VRAM with no separate
    /// device-local copy needed afterward. Not every device exposes it.
    DeviceLocalHostVisible,
}

/// A persistent, mapped staging area: memory the caller lands bytes into
/// once and reuses, rather than the fresh staging buffer `upload` allocates,
/// maps and frees on every call. See `Context::staging`.
///
/// NOT POOLED, AND NOT A CHANGE TO `upload`: a `Staging` is created by
/// `Context::staging` or `Context::readback_area` and destroyed by
/// `Context::destroy_staging`, and nothing in this file hands one out of a
/// pool. A recorded copy may name one (`record_copies_from_staging`,
/// `record_readback`), but nothing retires one into a batch. The caller owns
/// its lifetime and keeps it alive until every batch that copies it has
/// completed.
pub struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    bytes: u64,
    kind: StagingKind,
    /// A transfer destination too (`Context::readback_area`).
    readback: bool,
}

/// One region of `Context::record_copies_from_staging`: `bytes` from the
/// area at `staging_offset` into `dst` at `dst_offset`.
#[derive(Clone, Copy)]
pub struct StagingCopy<'a> {
    pub staging_offset: u64,
    pub dst: &'a Buffer,
    pub dst_offset: u64,
    pub bytes: u64,
}

/// One region of `Context::record_buffer_copies`: `bytes` from `src` at
/// `src_offset` into `dst` at `dst_offset`.
#[derive(Clone, Copy)]
pub struct BufferCopy<'a> {
    pub src: &'a Buffer,
    pub src_offset: u64,
    pub dst: &'a Buffer,
    pub dst_offset: u64,
    pub bytes: u64,
}

// R4: `ptr` is a raw pointer, so `Staging` is not auto-`Send` -- and the C
// ABI's `Inner` (src/ffi.rs) now holds a `HashMap<u32, Staging>` behind the
// same per-device `Mutex` that already guards its `Context` and its
// `HashMap<u32, Buffer>`, which requires `Inner: Send` for `Mutex<Inner>` to
// be `Sync`. This is sound for the same reason `Vec<T>`'s or `Box<T>`'s own
// raw pointer does not stop THEM being `Send`: a `Send` bound is about moving
// exclusive ownership across a thread boundary, not about concurrent shared
// access (that is `Sync`, deliberately not implemented here), and Rust's
// ownership rules mean at most one thread ever holds a given `Staging` at a
// time. `ptr` points into Vulkan-mapped, `HOST_COHERENT` memory that is valid
// for as long as the `Staging` exists on whichever thread currently owns it;
// nothing about crossing threads changes that.
unsafe impl Send for Staging {}

impl Staging {
    /// The mapping's size in bytes, as given to `Context::staging`.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Which memory this area was allocated from.
    pub fn kind(&self) -> StagingKind {
        self.kind
    }

    /// The whole mapping, writable. Every kind `Context::staging` produces is
    /// `HOST_COHERENT`, so a write here needs no flush before
    /// `Context::copy_from_staging` reads it.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes as usize) }
    }

    /// The whole mapping, for reading. A `readback_area` holds what
    /// `Context::record_readback` copied once the batch that carried the copy
    /// has completed. The memory is `HOST_COHERENT`, so no invalidate is needed.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes as usize) }
    }

    /// Whether `Context::record_readback` may copy into this area.
    pub fn is_readback(&self) -> bool {
        self.readback
    }

    /// The whole mapping's address, for a caller that needs the raw pointer
    /// rather than a slice -- an unbuffered file read
    /// (`FILE_FLAG_NO_BUFFERING`) landing straight into it, for one.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// This staging area as a kernel operand (R3): `Context::staging` now
    /// allocates `STORAGE_BUFFER | TRANSFER_SRC`, so the mapped area's
    /// `vk::Buffer` is a legal descriptor target, and a caller with the
    /// `DeviceLocalHostVisible` kind can bind the mapping itself and dispatch
    /// against it with no `copy_from_staging` in between at all. Pass the
    /// result to `Context::bind_operand` wherever `bind` would otherwise take
    /// a `&Buffer`.
    pub fn as_operand(&self) -> BufferRef {
        BufferRef {
            buffer: self.buffer,
            bytes: self.bytes,
            tracked: None,
        }
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
    /// R3: which of `bound`'s ids came from a `Staging` bound directly through
    /// `Context::bind_operand`, rather than a `Buffer` bound through either
    /// method -- `true` at index i means `check_dispatch` must not look
    /// `bound[i]` up in `Context::live_buffers` (an untracked operand; see
    /// `BufferRef::tracked`). Left empty by `bind` (and cleared by it, if a
    /// kernel was previously bound through `bind_operand`), which is why
    /// `bind` needed no change: an empty or short `bound_kinds` reads as
    /// "every slot is a `Buffer`", `bind`'s only possible meaning.
    bound_kinds: RefCell<Vec<bool>>,
    /// What `bind` recorded: the descriptor writes a dispatch applies to the
    /// set it takes. `bind` makes no Vulkan call, so the write happens at the
    /// dispatch, against that dispatch's own set.
    pending: RefCell<Vec<vk::DescriptorBufferInfo>>,
    /// The batch generation and set `pending` was last written into; cleared
    /// by every `bind`/`bind_operand`, so a hit always holds the current bindings.
    reuse: Cell<Option<(u64, vk::DescriptorSet)>>,
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
    use super::{pool_class, POOL_CLASSES_PER_OCTAVE, POOL_CLASS_FLOOR};

    #[test]
    fn a_pool_class_holds_its_request_within_an_eighth_and_is_monotonic() {
        assert!(POOL_CLASSES_PER_OCTAVE.is_power_of_two());
        let limit = u64::MAX;
        // exact at and below the floor: the word-rounded request itself
        for a in [4, 4096, 8192, POOL_CLASS_FLOOR - 4, POOL_CLASS_FLOOR] {
            assert_eq!(pool_class(a, limit), a, "{a}");
        }
        // above it: never smaller, a multiple of 4, under 1/8 larger, and
        // never smaller than the class of any smaller request
        let mut previous = pool_class(POOL_CLASS_FLOOR, limit);
        let mut a = POOL_CLASS_FLOOR + 4;
        while a < (1u64 << 34) {
            let c = pool_class(a, limit);
            assert!(c >= a, "{a} -> {c}");
            assert_eq!(c % 4, 0, "{a} -> {c}");
            assert!(
                (c - a) * POOL_CLASSES_PER_OCTAVE < a,
                "{a} -> {c}: more than 1/8"
            );
            assert!(c >= previous, "{a} -> {c} after {previous}");
            previous = c;
            // every size near an octave edge and a spread of sizes between
            a = if a < POOL_CLASS_FLOOR * 4 {
                a + 4
            } else {
                a + a / 997 + 4
            };
        }
        // the classes of one octave, 64 KiB..128 KiB: 8 KiB steps
        assert_eq!(pool_class(POOL_CLASS_FLOOR + 4, limit), 72 * 1024);
        assert_eq!(pool_class(1_000_000, limit), 1_048_576);
        assert_eq!(pool_class(1_040_000, limit), 1_048_576);
        assert_eq!(pool_class(1_048_580, limit), 1_179_648);
    }

    #[test]
    fn a_pool_class_never_crosses_the_device_limit() {
        // a request the device can hold is served at its own size rather than
        // refused because its class would not fit
        let limit = 1_000_000;
        assert_eq!(pool_class(999_996, limit), 999_996);
        assert_eq!(pool_class(900_000, limit), 917_504);
    }

    #[test]
    fn a_request_takes_its_own_class_first_then_the_smallest_larger_within_the_ratio() {
        use super::{pool_pick, POOL_REUSE_RATIO};
        use std::collections::BTreeMap;
        assert_eq!(POOL_REUSE_RATIO, 2);
        let mib = 1u64 << 20;
        let pool: BTreeMap<u64, ()> = [mib, 3 * mib / 2, 3 * mib, 48 * 1024]
            .into_iter()
            .map(|k| (k, ()))
            .collect();
        // its own class, whenever the pool holds one
        assert_eq!(pool_pick(&pool, mib, true), Some(mib));
        assert_eq!(pool_pick(&pool, 3 * mib, false), Some(3 * mib));
        // else the SMALLEST larger class, the ratio included
        assert_eq!(pool_pick(&pool, 5 * mib / 4, true), Some(3 * mib / 2));
        assert_eq!(pool_pick(&pool, mib / 2, true), Some(mib));
        assert_eq!(pool_pick(&pool, 3 * mib / 4, true), Some(mib));
        // never beyond the ratio, never a smaller class, never with reuse off
        assert_eq!(pool_pick(&pool, 3 * mib / 8, true), None);
        assert_eq!(pool_pick(&pool, 4 * mib, true), None);
        assert_eq!(pool_pick(&pool, 3 * mib / 4, false), None);
        // at or below the floor the key stays exact
        assert_eq!(pool_pick(&pool, 32 * 1024, true), None);
        assert_eq!(pool_pick(&pool, 48 * 1024, true), Some(48 * 1024));
        assert_eq!(pool_pick(&BTreeMap::<u64, ()>::new(), mib, true), None);
    }

    #[test]
    fn dispatch_timing_resolves_one_wrap_but_refuses_an_ambiguous_batch() {
        // An 8-bit clock at 2 ns/tick: these completion stamps cross zero.
        let intervals = super::dispatch_intervals(&[250, 3, 10], 8, 2.0, 100.0).unwrap();
        assert_eq!(
            intervals,
            vec![18, 14],
            "one interval per item, in item order"
        );
        assert_eq!(super::summarize_intervals(&intervals), (32, 14, 18));
        // Identical stamps cannot reveal an additional period of execution.
        assert!(matches!(
            super::dispatch_intervals(&[250, 3, 10], 8, 2.0, 512.0),
            Err(super::AckError::AmbiguousTiming(_))
        ));
        // a batch of no item has no interval and folds to the instrument's
        // "nothing timed" sentinel
        assert_eq!(super::summarize_intervals(&[]), (0, u64::MAX, 0));
    }

    #[test]
    fn each_interval_is_charged_to_its_own_items_tag_and_the_rows_partition_the_sum() {
        use super::{tag_rows_pending, tag_rows_resolved, DispatchTagTime};
        use std::collections::BTreeMap;
        let mut rows = BTreeMap::new();
        // a batch of five items: two tagged 7, one 3, one untagged, one 7
        let tags = [7u32, 7, 3, 0, 7];
        let intervals = [10u64, 25, 4, 6, 40];
        tag_rows_pending(&mut rows, &tags);
        assert_eq!(
            rows[&7].unresolved_items, 3,
            "every tagged item starts unresolved"
        );
        tag_rows_resolved(&mut rows, &tags, &intervals);
        assert_eq!(
            rows[&7],
            DispatchTagTime {
                tag: 7,
                items: 3,
                device_ns: 75,
                max_item_ns: 40,
                unresolved_items: 0
            },
            "the 1st, 2nd and 5th intervals, not the first three"
        );
        assert_eq!(rows[&3].device_ns, 4);
        assert_eq!(
            rows[&0].device_ns, 6,
            "untagged work keeps its own row, tag 0"
        );
        let total: u64 = rows.values().map(|r| r.device_ns).sum();
        assert_eq!(
            total,
            super::summarize_intervals(&intervals).0,
            "the rows partition the batch's span"
        );
        // a second batch that never resolves (its readback failed): its items
        // stay visible as unresolved, and the read rows do not move
        tag_rows_pending(&mut rows, &[3, 3]);
        assert_eq!(rows[&3].unresolved_items, 2);
        assert_eq!((rows[&3].items, rows[&3].device_ns), (1, 4));
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
