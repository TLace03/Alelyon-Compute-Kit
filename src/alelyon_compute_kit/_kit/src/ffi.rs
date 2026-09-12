//! C ABI for the Alelyon Compute Kit, so Python (ctypes) and the PyTorch
//! backend can open a device, move bytes, and run the BF16 matmul, the FP32
//! matmul (matmul-f32/v1), the strided pointwise family (pointwise-f32/v1),
//! the fixed-order reduction family (reduce-f32/v1), the row family
//! (rowwise-f32/v2: softmax, log-softmax and RMSNorm by row) and the casts
//! without knowing Vulkan. Explicit handles, explicit errors: every call returns
//! 0 on success or a negative code, and `ack_last_error` gives the message for
//! the calling thread.
//!
//! **Handles are ids, not addresses.** A device handle is a number from a
//! process-wide counter, and a buffer handle is `(device id << 32) | sequence`,
//! so every handle names its owner. Nothing here dereferences a handle: each
//! call looks it up in the registry under a lock and refuses, by name, a handle
//! that is closed (`ACK_ERR_CLOSED`), belongs to another device
//! (`ACK_ERR_FOREIGN`), or was freed or never allocated (`ACK_ERR_FREED`).
//! A device whose state is unknown after a failed fence wait refuses every
//! transfer, kernel and matmul call with `ACK_ERR_POISONED`; a free on it
//! consumes the handle, leaves the buffer allocated and returns that code
//! too; `ack_close` still succeeds on it and leaves every Vulkan object
//! allocated on purpose (ACK-L0-03).
//! Ids are sequential, not secrets: a forged handle equal to a live id is
//! indistinguishable from the real one, and that is not what this defends
//! against. It defends against use after close, use across devices, and use
//! after free, which were undefined behaviour at the C ABI before (review
//! findings ACK-FFI-01 and ACK-FFI-02).
//!
//! **Close is atomic.** `ack_close` looks the handle up, takes the device
//! lock, refuses while buffers are live, and otherwise marks the device closed
//! under that lock before removing it from the registry. The closed flag is
//! the atom: of two racing closes exactly one sets it and the other finds it,
//! and a call that fetched the device just before finds it when it takes the
//! lock. The registry lock is never held while waiting on a device. The Vulkan
//! objects go away when the last reference does, unless the device is
//! poisoned: then they are left allocated on purpose.
//!
//! **The ABI announces itself.** `ack_abi_info` reports the ABI version, the
//! struct size, the operator set with each operator's schema version, an
//! identity of the embedded shaders, a build identity, and which of the
//! capabilities a training SDK will need (streams, RNG, optimizer state,
//! serialization) exist, so a binding can refuse a stale library before
//! binding a single pointer and a contract can record what it ran on.
//! `ack_device_info` does the same for the opened device: PCI ids, device and
//! driver UUIDs, driver id and version, API version, features, the compute
//! limits the safe API checks against, what the matmul accepts, and one
//! identity hashed from all of it, so a checkpoint receipt can refuse a
//! resume on another device or driver before realising any device state.
//!
//! **A dispatch is recorded, not run** (increment 5). Since deferred
//! submission the operator entries record their dispatch into a batch the kit
//! submits at its next flush, so an entry returning `ACK_OK` means the
//! operation was ACCEPTED, not that it has finished. `ms_out` is NaN for the
//! same reason: no duration exists for the call to report. `ack_download` and
//! `ack_upload` are flush points, and so is `ack_close` after its live-buffer
//! refusal, so a readback sees every dispatch accepted before it OR IS
//! REFUSED: a flush that loses a batch of accepted dispatches poisons the
//! context, so the next call names the loss rather than returning bytes that
//! predate work the caller believes ran. A
//! failure of a flush is returned to whichever entry triggered it, with the
//! codes those entries already had (`-2`, `-10`), and its message names the
//! BATCH -- its reason, its item count and its sequence range -- because a
//! failure there cannot be attributed to one recorded operation. Every named
//! refusal in this file is a validation above any Vulkan call and is
//! unaffected in name, in code and in timing. See `ACK_ABI_VERSION` for what
//! this deliberately did not do, and the crate README for the flush list.
//!
//! The matmul is `C[M,N] = op(A)[M,K] . op(B)[K,N]` with BF16 operands and an
//! f32 result; `a_t` means A is stored as K rows of M, `b_t` means B is stored
//! as N rows of K. Shapes must be multiples of the 128x128 tile and K a
//! multiple of 32; anything else is refused (code -3), never padded silently.

use crate::batch::{Access, FlushReason};
use crate::embedding_ops::{EmbeddingOp, EmbeddingPlan, EmbeddingPlanError};
use crate::loss_ops::{LossOp, LossPlan, LossPlanError, LossReduction};
use crate::reduce_ops::{self, Finish, ReduceOp, ReducePlan, ReducePlanError, ReduceView};
use crate::row_ops::{RowOp, RowPlan, RowPlanError};
use crate::vq_ops::{VqPlan, VqPlanError};
use crate::{AckError, Buffer, Context, Kernel};
use std::cell::RefCell;

// a handle packs (device id, sequence) into a pointer-sized value, which is the
// contract of this ABI; a 32-bit target would truncate the owner half
#[cfg(not(target_pointer_width = "64"))]
compile_error!("the ACK C ABI packs (device id, sequence) into a pointer-sized handle and needs 64-bit pointers");
use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const SPV_NN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4_128x128_nn.spv");
const SPV_NT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4_128x128_nt.spv");
const SPV_TN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4_128x128_tn.spv");
const SPV_TT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4_128x128_tt.spv");
// The same kernel compiled with -DEDGE=1: bounds-checked staging and a
// predicated store, for shapes off the 128x128 tile. Separate modules, so the
// four above keep their exact bytes and an aligned shape keeps running the
// pipeline it ran before.
const SPV_E_NN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4e_128x128_nn.spv");
const SPV_E_NT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4e_128x128_nt.spv");
const SPV_E_TN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4e_128x128_tn.spv");
const SPV_E_TT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4e_128x128_tt.spv");
const SPV_CAST_F32_BF16: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");
const SPV_CAST_BF16_F32: &[u8] = include_bytes!("../kernels/cast_bf16_f32.spv");
const SPV_MATMUL_F32: &[u8] = include_bytes!("../kernels/matmul_f32.spv");
const SPV_MATMUL_F32_V2: &[u8] = include_bytes!("../kernels/matmul_f32_v2.spv");
const SPV_POINTWISE: &[u8] = include_bytes!("../kernels/pointwise_f32.spv");
// The same family over bf16 storage (schema 3), one switch per buffer.
// Indexed as `pointwise_ops::storage_module` - 1: the bits are x, y, z,
// out from the low end, and index 0 is SPV_POINTWISE above.
const SPV_POINTWISE_STORAGE: [&[u8]; 15] = [
    include_bytes!("../kernels/pointwise_bf16_x.spv"),
    include_bytes!("../kernels/pointwise_bf16_y.spv"),
    include_bytes!("../kernels/pointwise_bf16_xy.spv"),
    include_bytes!("../kernels/pointwise_bf16_z.spv"),
    include_bytes!("../kernels/pointwise_bf16_xz.spv"),
    include_bytes!("../kernels/pointwise_bf16_yz.spv"),
    include_bytes!("../kernels/pointwise_bf16_xyz.spv"),
    include_bytes!("../kernels/pointwise_bf16_out.spv"),
    include_bytes!("../kernels/pointwise_bf16_xout.spv"),
    include_bytes!("../kernels/pointwise_bf16_yout.spv"),
    include_bytes!("../kernels/pointwise_bf16_xyout.spv"),
    include_bytes!("../kernels/pointwise_bf16_zout.spv"),
    include_bytes!("../kernels/pointwise_bf16_xzout.spv"),
    include_bytes!("../kernels/pointwise_bf16_yzout.spv"),
    include_bytes!("../kernels/pointwise_bf16_xyzout.spv"),
];
const SPV_REDUCE: &[u8] = include_bytes!("../kernels/reduce_f32.spv");
const SPV_REDUCE_V2: &[u8] = include_bytes!("../kernels/reduce_f32_v2.spv");
/// The reduce family's bf16 storage modules (schema 2, 2026-09-10),
/// indexed by a PASS's own two bits minus one: 0 is bf16 in, 1 is bf16
/// out, 2 is both. Module 0 of each kernel is the f32 pair above, built
/// at open; these are built together on the first reduction that asks for
/// any of them, so a device that never sees a bf16 reduction builds none.
const SPV_REDUCE_STORAGE: [&[u8]; 3] = [
    include_bytes!("../kernels/reduce_bf16_in.spv"),
    include_bytes!("../kernels/reduce_bf16_out.spv"),
    include_bytes!("../kernels/reduce_bf16_inout.spv"),
];
const SPV_REDUCE_V2_STORAGE: [&[u8]; 3] = [
    include_bytes!("../kernels/reduce_v2_bf16_in.spv"),
    include_bytes!("../kernels/reduce_v2_bf16_out.spv"),
    include_bytes!("../kernels/reduce_v2_bf16_inout.spv"),
];
const SPV_ROWWISE: &[u8] = include_bytes!("../kernels/rowwise_f32.spv");
const SPV_VQ: &[u8] = include_bytes!("../kernels/vq4.spv");
pub const VQ_PUSH_BYTES: usize = crate::vq_ops::PUSH_BYTES;
const SPV_LOSS: &[u8] = include_bytes!("../kernels/loss_f32.spv");
const SPV_EMBEDDING: &[u8] = include_bytes!("../kernels/embedding_f32.spv");

use crate::matmul_ops;
use crate::pointwise_ops;
/// The coopmat kernels' BLOCKING FACTOR, reported as `matmul_tile` and
/// `matmul_k_multiple`. Since the `-DEDGE=1` variants these are NOT a shape
/// law: `ack_matmul_bf16` uses them to choose between the aligned and the edge
/// pipeline and to size the dispatch grid, and refuses only a zero extent. A
/// caller that reads them as "the shapes this device accepts" is reading them
/// wrongly, and no code in this repository does.
const TILE: u32 = 128;
const K_MULTIPLE: u32 = 32;
/// The coopmat modules' push block: M, N, K, zmode, za, zb, zc, kz, lda, ldb.
const COOPMAT_PUSH_BYTES: usize = 40;
/// Vulkan guarantees a z grid of at least 65,535 workgroups.
const MAX_GRID_Z: u32 = 65_535;
/// The aligned modules read A and B as uvec4 (eight bf16), so a per-z offset
/// must be a multiple of eight elements there; the edge modules gather.
const Z_STRIDE_MULTIPLE: u32 = 8;

/// The eight coopmat modules' block for one dispatch.
#[allow(clippy::too_many_arguments)]
fn coopmat_push(
    m: u32,
    n: u32,
    k: u32,
    zmode: u32,
    za: u32,
    zb: u32,
    zc: u32,
    kz: u32,
    lda: u32,
    ldb: u32,
) -> Vec<u8> {
    let mut push = Vec::with_capacity(COOPMAT_PUSH_BYTES);
    for word in [m, n, k, zmode, za, zb, zc, kz, lda, ldb] {
        push.extend_from_slice(&word.to_le_bytes());
    }
    push
}
const CAST_ELEMENTS_PER_GROUP: u32 = 256 * 4;

pub const ACK_OK: i32 = 0;
pub const ACK_ERR_NULL: i32 = -1;
pub const ACK_ERR_VULKAN: i32 = -2;
pub const ACK_ERR_SHAPE: i32 = -3;
pub const ACK_ERR_SIZE: i32 = -4;
/// `ack_close` while buffers are still allocated on the device (ACK-FFI-02).
pub const ACK_ERR_LIVE_BUFFERS: i32 = -5;
/// The device handle is not open: closed, never opened, or not a kit handle.
pub const ACK_ERR_CLOSED: i32 = -6;
/// The buffer handle belongs to a different device than the one named.
pub const ACK_ERR_FOREIGN: i32 = -7;
/// The buffer handle is unknown to its device: freed, or never allocated.
pub const ACK_ERR_FREED: i32 = -8;
/// The caller's `AckAbiInfo` is smaller than this library's.
pub const ACK_ERR_ABI: i32 = -9;
/// The device's state is unknown after a failed wait, or the device is lost;
/// every call on it refuses until it is closed (ACK-L0-03).
pub const ACK_ERR_POISONED: i32 = -10;
/// The operator's kernels could not be built on this device (the BF16
/// cooperative-matrix product on a device without the extension); the other
/// operators on the device still work.
pub const ACK_ERR_UNSUPPORTED_OP: i32 = -11;
/// A timed dispatch completed but its duration could not be resolved (the
/// batch ran at or beyond the timestamp counter's period, ACK-L0-04). The
/// matmul and cast entries never returned it: the product is done, so they
/// returned `ACK_OK` with `ms_out` set to NaN and the reason in
/// `ack_last_error`.
///
/// SINCE DEFERRED SUBMISSION NO ENTRY IN THIS FILE CAN PRODUCE IT AT ALL: the
/// operator entries record their dispatch instead of timing it, so no C entry
/// calls `Context::dispatch_timed` and nothing here resolves a timestamp. The
/// code stays declared -- removing it would be an ABI change, and the Rust
/// probes still reach `dispatch_timed` and still get `AmbiguousTiming` from it
/// -- but a caller of this ABI will not see it.
pub const ACK_ERR_TIMING: i32 = -12;

/// Bumped whenever a signature, a code, or a struct in this file changes.
/// 2: `AckAbiInfo` grew the schema and capability fields, `ack_device_info`
/// and `AckDeviceInfo` were added (2026-09-06).
/// 3: `ACK_ERR_POISONED` (-10) was added, and `ack_buffer_free` returns it for
/// a buffer left allocated on a poisoned device (2026-09-06, ACK-L0-03).
/// 4: `ack_matmul_f32` (matmul-f32/v1, operator bit 4, schema 1) and
/// `AckAbiInfo::op_schema_matmul_f32` were added, `ack_open` no longer
/// requires BF16 cooperative matrices, and `ack_matmul_bf16` returns
/// `ACK_ERR_UNSUPPORTED_OP` (-11) on a device without them (2026-09-06).
/// 5: `ack_live_objects` and `ack_leaked_on_poison` were added, the device
/// record's feature bits gained `ACK_FEATURE_SHADER_INT16`, `ack_open`
/// refuses a device without `shaderInt16`, and `ACK_ERR_TIMING` (-12) names
/// an unresolvable dispatch duration (2026-09-07, ACK-L0-05 export, the
/// validation layer's Int16 finding, ACK-L0-04).
/// 6: `ack_pointwise` (pointwise-f32/v1, operator bit 8, schema 1) and
/// `AckAbiInfo::op_schema_pointwise` were added, with `op_schema_reduce`,
/// `op_schema_rowwise`, `op_schema_loss` and `op_schema_embedding` reserved
/// at 0 (absent) so the later operator families append no further relayout;
/// the shader identity covers eight modules (2026-09-07). Still 6 when the
/// pointwise schema moved to 2 (operations 6-23 as new values of the
/// operation word; the block, buffers and entry are unchanged, 2026-09-07).
/// 7: `ack_reduce` (reduce-f32/v1, operator bit 16, schema 1) was added and
/// `AckAbiInfo::op_schema_reduce` reports it; the shader identity covers nine
/// modules (2026-09-07).
/// 8: `ack_rowwise` (rowwise-f32/v2 at the C ABI, operator bit 32, schema 1)
/// was added and `AckAbiInfo::op_schema_rowwise` reports it; the shader
/// identity covers ten modules (2026-09-07).
/// 9: `ack_loss` (indexed-cross-entropy-f32/v2 at the C ABI, operator bit 64,
/// schema 1) was added and `AckAbiInfo::op_schema_loss` reports it; the shader
/// identity covers eleven modules (2026-09-07). Still 9 when `ack_embedding`
/// (embedding-f32/v2, operator bit 128, schema 1) was added and
/// `AckAbiInfo::op_schema_embedding` began reporting it: no field moved, and
/// one rebuild-forcing bump serves both families (2026-09-08).
///
/// STILL 9 WHEN DEFERRED SUBMISSION LANDED (increment 5, 2026-09-08), AND
/// THAT IS FLAGGED RATHER THAN SETTLED. The rule above is about signatures,
/// codes and structs, and none of the three moved: every entry keeps its
/// signature, every code keeps its meaning, no struct changed a field. What
/// DID change is behaviour behind unchanged signatures, and it is named here
/// so nobody has to rediscover it:
///
/// - the operator entries RECORD their dispatch and it runs at the next flush,
///   so `ms_out` is NaN where it used to be a duration (the value `ms_out`
///   already carried for a completed operation whose timing was unresolvable);
/// - `ACK_ERR_TIMING` (-12) is no longer producible by any entry here;
/// - `ack_buffer_free` returning `ACK_OK` means the handle is consumed and the
///   objects are destroyed at the next flush, not that they are already gone;
/// - `ack_buffer_free` can also return `-2` or `-10` for the RETIREMENT FLUSH
///   it triggered rather than for the buffer, naming a batch some earlier call
///   filled. Same two codes, wider circumstances. The alternative was
///   discarding the flush's result, which is a swallowed device error;
/// - `ack_download` and `ack_upload` are flush points, and `ack_close` flushes
///   after its live-buffer refusal;
/// - a failure of a flush is reported to whichever entry triggered it and
///   names the batch, not one operation;
/// - a flush that loses a batch of already-accepted dispatches POISONS the
///   context, so every later entry returns `-10` by name rather than serving
///   a readback of bytes that predate work the caller was told had been
///   accepted.
///
/// TWO THINGS THIS DELIBERATELY DID NOT DO, both because they need the ABI
/// owner and the other lane at ABI 9:
///
/// 1. `ack_sync(dev) -> i32` was NOT added. Without it the torch adapter's four
///    in-flight refusers (`ack_snapshot`, `ack_counters_reset`,
///    `ack_set_f32_mm_mode`, `ack_set_bmm_bf16_on_kit`) have no way to reach a
///    kit flush -- none of them makes a kit call at all -- so
///    `batch::FlushReason::Accounting` is declared and UNWIRED and the hazard
///    it names is real and unguarded: an accounting frame taken with work
///    recorded and unsubmitted counts operations that might still fail, and a
///    generation change lets work started in generation g finish in g+1.
/// 2. `ack_close` does NOT return a flush failure; it puts it in
///    `ack_last_error` and still returns `ACK_OK`, because today it always
///    succeeds once the buffers are gone and that is pinned.
///
/// 10: `ack_matmul_bf16_z` (the coopmat product with a z grid: a batch of
/// products through per-z operand offsets, or a K-split writing partials;
/// operator bit 1, schema 2) was added and `AckAbiInfo::op_schema_matmul_bf16`
/// moved to 2; `ack_matmul_bf16` keeps its signature and its results, its
/// eight modules carry the z words with zeros for a single product
/// (2026-09-09).
pub const ACK_ABI_VERSION: u32 = 10;
/// Operator bits reported by `ack_abi_info`.
pub const ACK_OP_MATMUL_BF16: u64 = 1;
pub const ACK_OP_CAST: u64 = 2;
/// Schema version of each operator's C signature and semantics.
/// 2 since ABI 10: `ack_matmul_bf16_z` exists beside `ack_matmul_bf16`.
pub const ACK_OP_SCHEMA_MATMUL_BF16: u32 = 2;
pub const ACK_OP_SCHEMA_CAST: u32 = 1;
/// The FP32 product family: `ack_matmul_f32` with the 64-byte plan block.
pub const ACK_OP_MATMUL_F32: u64 = 4;
/// Schema 2 (2026-09-09): kinds 2 and 3 run the plain-fp32 tiled kernel
/// matmul-f32/v2 through the SAME entry and block; kinds 0 and 1 are v1,
/// unchanged. A new kind value is a schema move, not an ABI move, exactly as
/// pointwise schema 2 was: no signature, code or struct changed.
pub const ACK_OP_SCHEMA_MATMUL_F32: u32 = matmul_ops::SCHEMA;
/// The strided pointwise family: `ack_pointwise` with the 120-byte plan block
/// (pointwise-f32/v1, ABI 6). Schema 1 carried operations 0-5; schema 2
/// (2026-09-07) carries 0-23 in the same block, so a binding written for
/// schema 1 that meets this library is refused by its own pin, never handed
/// an operation word it cannot name.
pub const ACK_OP_POINTWISE: u64 = 8;
pub const ACK_OP_SCHEMA_POINTWISE: u32 = pointwise_ops::SCHEMA;
/// The fixed-order reduction family: `ack_reduce` with the 80-byte request
/// block (reduce-f32/v1, ABI 7).
pub const ACK_OP_REDUCE: u64 = 16;
/// 2 since 2026-09-10: `ack_reduce` also accepts a request block of
/// `reduce_ops::REQUEST_BYTES` -- the same 80-byte push layout followed by
/// one STORAGE FLAGS word (bit 0 the input buffer holds bf16, bit 1 the
/// output does). The 80-byte block still means f32 on both sides and is
/// bit-for-bit the call it always was, no signature or struct changed, and
/// the kernel's own push block is untouched: the flags word never reaches
/// a shader, it selects which shader.
pub const ACK_OP_SCHEMA_REDUCE: u32 = 2;
/// The row family: `ack_rowwise` with the kernel's 16-byte push block
/// (rowwise-f32/v2, ABI 8).
pub const ACK_OP_ROWWISE: u64 = 32;
pub const ACK_OP_SCHEMA_ROWWISE: u32 = 1;
/// The indexed cross-entropy family: `ack_loss` with the kernel's 20-byte push
/// block, whose targets the entry validates and uploads
/// (indexed-cross-entropy-f32/v2, ABI 9).
pub const ACK_OP_LOSS: u64 = 64;
pub const ACK_OP_SCHEMA_LOSS: u32 = 1;
/// The embedding family: `ack_embedding`, whose index and offset images the
/// entry derives and uploads (embedding-f32/v2, ABI 9).
pub const ACK_OP_EMBEDDING: u64 = 128;
pub const ACK_OP_SCHEMA_EMBEDDING: u32 = 1;
/// The embedding family's push block: n, vocab, dim, operation.
pub const EMBEDDING_PUSH_BYTES: usize = 16;
/// The loss family's push block: rows, cols, operation, reduction, normalizer.
pub const LOSS_PUSH_BYTES: usize = 20;
/// The row family's request block: rows, cols, operation, epsilon bits.
pub const ROWWISE_PUSH_BYTES: usize = 16;
/// Version of `AckDeviceInfo`.
pub const ACK_DEVICE_INFO_VERSION: u32 = 1;
/// Version of `AckDispatchStats`.
pub const ACK_DISPATCH_STATS_VERSION: u32 = 1;

/// Feature bits in `AckDeviceInfo::features`.
pub const ACK_FEATURE_COOPERATIVE_MATRIX: u64 = 1;
pub const ACK_FEATURE_BF16_TYPE: u64 = 2;
pub const ACK_FEATURE_BF16_DOT_PRODUCT: u64 = 4;
pub const ACK_FEATURE_BF16_COOPERATIVE_MATRIX: u64 = 8;
pub const ACK_FEATURE_SUBGROUP_SIZE_CONTROL: u64 = 16;
/// The 16x16x16 bf16 x bf16 -> f32 subgroup-scope cooperative shape the
/// matmul kernels are written for is advertised by the driver.
pub const ACK_FEATURE_COOPMAT_BF16_16X16X16_SUBGROUP: u64 = 32;
/// `shaderInt16` is enabled on the device (the f32 -> bf16 cast kernel
/// declares Int16; `ack_open` refuses a device without it, so an open device
/// always has it).
pub const ACK_FEATURE_SHADER_INT16: u64 = 64;
/// Dtype codes in `AckDeviceInfo`.
pub const ACK_DTYPE_BF16: u32 = 1;
pub const ACK_DTYPE_F32: u32 = 2;

/// What `ack_abi_info` fills in. `size` is this library's size of the struct,
/// so a caller compiled against a larger later version can see the difference;
/// a caller must require `size` to equal its own struct's size.
#[repr(C)]
pub struct AckAbiInfo {
    pub size: u32,
    pub version: u32,
    pub operators: u64,
    /// FNV-1a over the ten embedded SPIR-V modules: the shader identity of this
    /// library, so two libraries with the same version but different kernels
    /// are told apart. An identity, not a proof of what the kernels compute.
    pub shaders: u64,
    /// FNV-1a over the crate version and the shader identity: which build this is.
    pub build: u64,
    pub op_schema_matmul_bf16: u32,
    pub op_schema_cast: u32,
    /// Capability versions a training SDK asks for; 0 means the capability
    /// does not exist in this library, and a consumer must treat 0 as absent,
    /// never as "version 0 of something".
    pub capability_streams: u32,
    pub capability_rng: u32,
    pub capability_optimizer: u32,
    pub capability_serialization: u32,
    /// Schema of `ack_matmul_f32` (matmul-f32/v1); appended in ABI 4 so the
    /// earlier fields keep their offsets.
    pub op_schema_matmul_f32: u32,
    /// Schema of `ack_pointwise` (pointwise-f32/v1); appended in ABI 6.
    pub op_schema_pointwise: u32,
    /// Schema of `ack_reduce` (reduce-f32/v1); reserved in ABI 6, set in ABI 7.
    pub op_schema_reduce: u32,
    /// Schema of `ack_rowwise` (rowwise-f32/v2 at the C ABI); reserved in
    /// ABI 6, set in ABI 8.
    pub op_schema_rowwise: u32,
    /// Schema of `ack_loss` (indexed-cross-entropy-f32/v2 at the C ABI);
    /// reserved in ABI 6, set in ABI 9.
    pub op_schema_loss: u32,
    /// Schema of `ack_embedding` (embedding-f32/v2); reserved in ABI 6, set
    /// alongside the loss family under ABI 9.
    pub op_schema_embedding: u32,
}

/// What the dispatch instrument has counted and timed (`ack_dispatch_stats`).
///
/// SELF-DESCRIBING, AND THAT IS WHY THE ABI VERSION DID NOT MOVE FOR IT. The
/// version rule above is about signatures, codes and structs that a caller's
/// pointer table already binds; this entry is in none of them. It is not in
/// `torch_backend/ack_backend.cpp`'s table, it is resolved by name at the
/// Python edge where its absence is tolerated, and it changes no existing
/// signature, code or field. What replaces the bump is this struct's own
/// `size` and `version`, which a reader must require to equal its own -- the
/// handshake `AckDeviceInfo` already uses. A library that later changes this
/// layout is refused by name by a caller written for the old one.
///
/// `device_ns` is DEVICE-ELAPSED TIME summed over recorded items: barrier
/// drain, wave launch, kernel arithmetic and memory traffic together. Wall time outside this span also includes device transfers and queue waits. It is not a utilisation figure and not a
/// split of arithmetic from memory traffic. It is 0, with
/// `timestamps_enabled` 0, unless `ACK_DISPATCH_TIMESTAMPS` was set when the
/// device was opened -- so an absent figure is distinguishable from a measured
/// zero, which a bare 0 would not be.
#[repr(C)]
pub struct AckDispatchStats {
    pub size: u32,
    pub version: u32,
    /// Items recorded by the deferred batch recorder.
    pub items: u64,
    /// `vkCmdDispatch` commands recorded; `>= items`.
    pub commands: u64,
    /// Pipeline barriers recorded, including the instrument's batch-origin
    /// dependency when enabled.
    pub barriers: u64,
    /// Batches submitted, whatever the flush reason.
    pub flushes: u64,
    /// Flushes whose timestamps were read back.
    pub timed_flushes: u64,
    /// Items covered by `device_ns`.
    pub timed_items: u64,
    /// Device-elapsed nanoseconds over `timed_items`.
    pub device_ns: u64,
    /// Shortest and longest single item's device-elapsed time.
    pub min_item_ns: u64,
    pub max_item_ns: u64,
    /// Flushes whose span could NOT be folded in. Nonzero means `device_ns`
    /// covers less than the run did.
    pub unresolved_flushes: u64,
    /// 1 when the timestamp instrument was on at open, 0 otherwise.
    pub timestamps_enabled: u32,
    pub reserved: u32,
}

/// What `ack_device_info` fills in for an open device. Fixed-width, `repr(C)`,
/// no pointers: a contract can store the bytes as they are. `identity` is
/// FNV-1a over every numeric field of this struct (bytes 8 to 160: the PCI
/// ids, both UUIDs, driver id and version, API version, features, limits and
/// the matmul contract) followed by the library's shader identity; the names
/// are not hashed (they are display text). The same device, driver, reported
/// capabilities and kernels give the same value; a change in any of them
/// changes it. It identifies; it proves nothing about what the device
/// computes, and cross-process stability rests on Vulkan's declaration that
/// `deviceUUID` and `driverUUID` are stable for a device and a driver build.
#[repr(C)]
pub struct AckDeviceInfo {
    pub size: u32,
    pub version: u32,
    pub api_version: u32,
    pub driver_version_raw: u32,
    pub driver_id: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
    pub features: u64,
    pub subgroup_size: u32,
    pub timestamp_valid_bits: u32,
    pub timestamp_period_ns: f32,
    pub max_push_constant_bytes: u32,
    pub device_local_bytes: u64,
    pub max_storage_buffer_bytes: u64,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_size: [u32; 3],
    pub matmul_tile: u32,
    pub matmul_k_multiple: u32,
    pub matmul_layouts: u32,
    pub matmul_operand_dtype: u32,
    pub matmul_result_dtype: u32,
    pub cast_modes: u32,
    pub reserved: u32,
    pub identity: u64,
    pub device_name: [c_char; 256],
    pub driver_name: [c_char; 64],
    pub driver_info: [c_char; 128],
}

/// Copy `src` into a fixed C string, truncating on a UTF-8 character boundary
/// (Vulkan allows 256-byte names; the driver fields here are shorter, so a
/// long `driverInfo` is cut, never split inside a multi-byte character).
fn fill_c_string(dst: &mut [c_char], src: &str) {
    let mut n = src.len().min(dst.len() - 1);
    while n > 0 && !src.is_char_boundary(n) {
        n -= 1;
    }
    let bytes = src.as_bytes();
    for (d, &b) in dst.iter_mut().zip(bytes[..n].iter()) {
        *d = b as c_char;
    }
    dst[n] = 0;
}

/// Byte range of `AckDeviceInfo` the identity hashes: after `size`/`version`,
/// up to `identity` itself.
const DEVICE_INFO_HASHED: std::ops::Range<usize> = 8..160;

/// The identity of a filled `AckDeviceInfo`: its numeric fields and the
/// library's shaders (see the struct doc).
pub fn info_identity(info: &AckDeviceInfo) -> u64 {
    // a repr(C) struct with no padding in this range (pinned by the offset
    // test below) reads as plain bytes
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (info as *const AckDeviceInfo).cast::<u8>(),
            std::mem::size_of::<AckDeviceInfo>(),
        )
    };
    fnv1a(&[&bytes[DEVICE_INFO_HASHED], &shader_identity().to_le_bytes()])
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = message.into());
}

fn code_for(err: &AckError) -> i32 {
    match err {
        AckError::Unsupported(_) => ACK_ERR_SHAPE,
        AckError::Foreign(_) => ACK_ERR_FOREIGN,
        AckError::Freed(_) => ACK_ERR_FREED,
        AckError::Poisoned(_) => ACK_ERR_POISONED,
        AckError::AmbiguousTiming(_) => ACK_ERR_TIMING,
        _ => ACK_ERR_VULKAN,
    }
}

/// Report the duration of a RECORDED dispatch: there is none.
///
/// Since deferred submission (increment 5) the operator entries RECORD their
/// dispatch and it is submitted at the next flush, so no device duration
/// exists for the call to report. `ms_out` gets NaN, which is exactly the
/// signal these entries already used for "the operation is accepted, only its
/// duration is unknown" (ACK-L0-04's `AmbiguousTiming`, `ack_matmul_bf16` and
/// the rest). No signature changed and no entry gained or lost a code.
///
/// No message is set on this path, on purpose: the reason is a constant, and
/// formatting one into `ack_last_error` on every accepted dispatch would be a
/// heap allocation per dispatch in the one path deferred submission exists to
/// make cheap. The reason is in these docs and in the crate README instead.
///
/// # Safety
/// `ms_out` must be null or point to a writable f64.
unsafe fn no_duration(ms_out: *mut f64) {
    if !ms_out.is_null() {
        unsafe { *ms_out = f64::NAN };
    }
}

/// A `Context::dispatch` result at the C ABI: `ACK_OK` for a recorded
/// dispatch, or the code and message of whatever refused. A refusal that
/// belongs to the deferred batch rather than to this call says so in its
/// message (see `Context::flush`).
fn recorded(r: crate::Result<()>) -> i32 {
    match r {
        Ok(()) => ACK_OK,
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Shader identity of this library: FNV-1a over the embedded SPIR-V, in the
/// order NN, NT, TN, TT, edge NN, edge NT, edge TN, edge TT, f32->bf16,
/// bf16->f32, matmul-f32, pointwise-f32, reduce-f32, rowwise-f32, loss-f32,
/// embedding-f32.
///
/// The four edge modules are hashed here because this value exists to tell two
/// libraries of the same version apart by what their kernels compute, and a
/// library that can run an off-tile product is not the one that refused it.
/// Admitting them CHANGES this identity and therefore `build_identity`, so a
/// checkpoint receipt written by an earlier build refuses to resume against
/// this one. That is a declared consequence of the change, not a surprise.
pub fn shader_identity() -> u64 {
    fnv1a(&[
        SPV_NN,
        SPV_NT,
        SPV_TN,
        SPV_TT,
        SPV_E_NN,
        SPV_E_NT,
        SPV_E_TN,
        SPV_E_TT,
        SPV_CAST_F32_BF16,
        SPV_CAST_BF16_F32,
        SPV_MATMUL_F32,
        SPV_MATMUL_F32_V2,
        SPV_POINTWISE,
        SPV_REDUCE,
        SPV_REDUCE_V2,
        SPV_ROWWISE,
        SPV_LOSS,
        SPV_EMBEDDING,
        SPV_VQ,
    ])
}

/// Build identity of this library: the crate version and the shader identity.
pub fn build_identity() -> u64 {
    fnv1a(&[
        env!("CARGO_PKG_VERSION").as_bytes(),
        &shader_identity().to_le_bytes(),
    ])
}

/// The identity of the device alone (its report's identity bytes and the
/// shaders): the coarser of the two identities, kept for the safe-runtime
/// tests; `AckDeviceInfo::identity` also covers features and limits.
pub fn device_identity(report: &crate::DeviceReport) -> u64 {
    fnv1a(&[&report.identity_bytes(), &shader_identity().to_le_bytes()])
}

// ---- handles ------------------------------------------------------------------

/// A buffer handle names its owner in the high 32 bits and the buffer in the
/// low 32; neither half is ever zero for a live buffer.
pub fn buffer_handle(device_id: u64, seq: u32) -> u64 {
    (device_id << 32) | seq as u64
}

/// The (owner, sequence) pair a buffer handle encodes.
pub fn split_buffer_handle(handle: u64) -> (u64, u32) {
    (handle >> 32, (handle & 0xffff_ffff) as u32)
}

/// What one open device owns. Everything Vulkan sits behind the device's
/// mutex (`AckDevice::inner`), so `Context` may be `Send` and not `Sync` and
/// two threads cannot interleave on one queue (ACK-FFI-01).
struct Inner {
    ctx: Context,
    kernels: Vec<Kernel>, // aligned then edge NN, NT, TN, TT (edge*4 + a_t*2 + b_t); empty without BF16 coopmat
    casts: Vec<Kernel>,   // f32->bf16, bf16->f32
    /// matmul-f32/v1 (three storage buffers, a 64-byte plan block); `Option`
    /// only so `Drop` can take it.
    matmul_f32: Option<Kernel>,
    /// matmul-f32/v2, the plain-fp32 tiled kernel kinds 2 and 3 bind (the same
    /// buffers and block as v1); the same `Option` for `Drop`.
    matmul_f32_v2: Option<Kernel>,
    /// pointwise-f32/v1 (four storage buffers, a 120-byte plan block); the
    /// same `Option` for `Drop`.
    pointwise: Option<Kernel>,
    /// The same family's bf16 storage modules (schema 3), built on the
    /// first call that asks for one rather than at open: a device that
    /// never runs a bf16 operand never builds them, and no error path in
    /// the open sequence moves.
    pointwise_storage: [Option<Kernel>; 15],
    /// reduce-f32/v1 (two storage buffers, an 80-byte push block); the same.
    reduce: Option<Kernel>,
    /// reduce-f32/v2 (2026-09-09; the same two buffers, a 96-byte push block):
    /// the kept-across-the-lanes kernel `ReducePlan` selects for a first pass
    /// whose kept side owns the unit stride; the same `Option` for `Drop`.
    reduce_v2: Option<Kernel>,
    /// The bf16 storage modules of each reduce kernel, built together on
    /// the first reduction whose flags word is non-zero.
    reduce_storage: [Option<Kernel>; 3],
    reduce_v2_storage: [Option<Kernel>; 3],
    /// `ACK_REDUCE_V1_ONLY` was set when this device opened: every reduce
    /// plan keeps v1 (the measured A/B control; not a contract).
    reduce_v1_only: bool,
    /// rowwise-f32/v2 (four storage buffers, a 16-byte push block); the same.
    rowwise: Option<Kernel>,
    /// indexed-cross-entropy-f32/v2 (four storage buffers, a 20-byte push
    /// block); the same.
    loss: Option<Kernel>,
    /// embedding-f32/v2 (four storage buffers, a 16-byte push block); the same.
    embedding: Option<Kernel>,
    /// Optional vq4/v1 extension. Built only after a request passes live
    /// buffer admission, retired with the same context as every kernel.
    vq: Option<Kernel>,
    buffers: HashMap<u32, Buffer>,
    next_buffer: u32,
    /// Set by `ack_close` under the lock; every later call refuses by name.
    closed: bool,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        // A device dropped without `ack_close` (the registry going away at
        // process teardown) may still hold a recorded, unsubmitted batch.
        // Discard it FIRST: its commands name descriptor sets of the kernels
        // this loop is about to destroy, and nothing was submitted, so
        // dropping them is safe. `ack_close` flushes instead -- work a caller
        // was told was accepted is not silently discarded on that path.
        ctx.discard_recorded_batch();
        for (_, b) in self.buffers.drain() {
            ctx.destroy_buffer(b);
        }
        for k in self.kernels.drain(..) {
            ctx.destroy_kernel(k);
        }
        for k in self.casts.drain(..) {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.matmul_f32.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.matmul_f32_v2.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.pointwise.take() {
            ctx.destroy_kernel(k);
        }
        for slot in self.pointwise_storage.iter_mut() {
            if let Some(k) = slot.take() {
                ctx.destroy_kernel(k);
            }
        }
        if let Some(k) = self.reduce.take() {
            ctx.destroy_kernel(k);
        }
        for slot in self
            .reduce_storage
            .iter_mut()
            .chain(self.reduce_v2_storage.iter_mut())
        {
            if let Some(k) = slot.take() {
                self.ctx.destroy_kernel(k);
            }
        }
        if let Some(k) = self.reduce_v2.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.rowwise.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.loss.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.embedding.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.vq.take() {
            ctx.destroy_kernel(k);
        }
    }
}

/// An open device with the FP32 matmul, the pointwise family, the reduction
/// family, the row family and the two casts built, and the four BF16 matmul
/// layouts where the device has BF16 cooperative matrices.
pub struct AckDevice {
    id: u64,
    inner: Mutex<Inner>,
}

struct Registry {
    devices: HashMap<u64, Arc<AckDevice>>,
    next_device: u64,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn registry() -> MutexGuard<'static, Registry> {
    let m = REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            devices: HashMap::new(),
            next_device: 1,
        })
    });
    // the registry is only ever inserted into, looked up, or removed from;
    // a panic elsewhere cannot leave it half-updated, so recover the data
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Look a device handle up. Never dereferences it.
fn device(handle: *const AckDevice) -> Result<Arc<AckDevice>, i32> {
    let id = handle as usize as u64;
    if id == 0 {
        return Err(ACK_ERR_NULL);
    }
    match registry().devices.get(&id) {
        Some(dev) => Ok(Arc::clone(dev)),
        None => {
            set_error(format!(
                "device handle {id} is not open (closed, never opened, or not a kit handle)"
            ));
            Err(ACK_ERR_CLOSED)
        }
    }
}

/// The device's lock, or a named refusal once it is closed. A poisoned lock
/// means a previous holder panicked mid-call with the Vulkan state unknown;
/// nothing after that is safe to run, so the process stops rather than guess.
fn inner(dev: &AckDevice) -> Result<MutexGuard<'_, Inner>, i32> {
    let guard = dev.inner.lock().unwrap_or_else(|_| {
        set_error("device lock poisoned by an earlier panic; the process cannot continue safely");
        std::process::abort()
    });
    if guard.closed {
        set_error(format!("device {} is closed", dev.id));
        return Err(ACK_ERR_CLOSED);
    }
    Ok(guard)
}

/// A buffer handle resolved against the device that must own it.
fn buffer_of<'a>(
    dev: &AckDevice,
    inner: &'a Inner,
    handle: *const Buffer,
) -> Result<&'a Buffer, i32> {
    let raw = handle as usize as u64;
    if raw == 0 {
        return Err(ACK_ERR_NULL);
    }
    let (owner, seq) = split_buffer_handle(raw);
    if owner != dev.id {
        set_error(format!(
            "buffer {raw:#x} belongs to device {owner}, not device {}",
            dev.id
        ));
        return Err(ACK_ERR_FOREIGN);
    }
    inner.buffers.get(&seq).ok_or_else(|| {
        set_error(format!(
            "buffer {raw:#x} is not live on device {} (freed, or never allocated)",
            dev.id
        ));
        ACK_ERR_FREED
    })
}

// ---- entry points --------------------------------------------------------------

/// Fill `out` (of `size` bytes) with this library's ABI description. A caller
/// checks `version` against the one it was written for before binding any
/// other symbol; a struct smaller than this library's is refused.
/// # Safety
/// `out` must point to `size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_abi_info(out: *mut AckAbiInfo, size: usize) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    if size < std::mem::size_of::<AckAbiInfo>() {
        set_error(format!(
            "AckAbiInfo of {size} bytes is smaller than this library's {}",
            std::mem::size_of::<AckAbiInfo>()
        ));
        return ACK_ERR_ABI;
    }
    let info = AckAbiInfo {
        size: std::mem::size_of::<AckAbiInfo>() as u32,
        version: ACK_ABI_VERSION,
        operators: ACK_OP_MATMUL_BF16
            | ACK_OP_CAST
            | ACK_OP_MATMUL_F32
            | ACK_OP_POINTWISE
            | ACK_OP_REDUCE
            | ACK_OP_ROWWISE
            | ACK_OP_LOSS
            | ACK_OP_EMBEDDING,
        shaders: shader_identity(),
        build: build_identity(),
        op_schema_matmul_bf16: ACK_OP_SCHEMA_MATMUL_BF16,
        op_schema_cast: ACK_OP_SCHEMA_CAST,
        capability_streams: 0,
        capability_rng: 0,
        capability_optimizer: 0,
        capability_serialization: 0,
        op_schema_matmul_f32: ACK_OP_SCHEMA_MATMUL_F32,
        op_schema_pointwise: ACK_OP_SCHEMA_POINTWISE,
        op_schema_reduce: ACK_OP_SCHEMA_REDUCE,
        op_schema_rowwise: ACK_OP_SCHEMA_ROWWISE,
        op_schema_loss: ACK_OP_SCHEMA_LOSS,
        op_schema_embedding: ACK_OP_SCHEMA_EMBEDDING,
    };
    // the caller promised `size` writable bytes, not alignment: write unaligned
    unsafe { std::ptr::write_unaligned(out, info) };
    ACK_OK
}

/// Open the best device and build its kernels. Returns a handle, or null with
/// `ack_last_error` set. The handle is an id: it must be closed with `ack_close`
/// and is refused, not dereferenced, after that.
#[no_mangle]
pub extern "C" fn ack_open() -> *mut AckDevice {
    let ctx = match Context::open() {
        Ok(c) => c,
        Err(e) => {
            set_error(format!("{e}"));
            return std::ptr::null_mut();
        }
    };
    // the BF16 cooperative-matrix kernels need subgroup size control and the
    // extension; a device without them opens with the FP32 product and the
    // casts only, and `ack_matmul_bf16` refuses on it by name (ABI 4)
    let bf16_capable = ctx.subgroup_size_control && ctx.report.bf16_cooperative_matrix;
    let mut kernels: Vec<Kernel> = Vec::with_capacity(8);
    if bf16_capable {
        // aligned NN, NT, TN, TT first, then the four edge variants at +4:
        // `ack_matmul_bf16` indexes this vector as edge*4 + a_t*2 + b_t
        for spv in [
            SPV_NN, SPV_NT, SPV_TN, SPV_TT, SPV_E_NN, SPV_E_NT, SPV_E_TN, SPV_E_TT,
        ] {
            // the 32-byte push block: M, N, K and the five z words (ABI 10)
            match ctx.kernel_with_subgroup(spv, 3, COOPMAT_PUSH_BYTES as u32, Some(64)) {
                Ok(k) => kernels.push(k),
                Err(e) => {
                    set_error(format!("matmul pipeline: {e}"));
                    for k in kernels.drain(..) {
                        ctx.destroy_kernel(k);
                    }
                    return std::ptr::null_mut();
                }
            }
        }
    }
    let mut casts: Vec<Kernel> = Vec::with_capacity(2);
    for spv in [SPV_CAST_F32_BF16, SPV_CAST_BF16_F32] {
        match ctx.kernel(spv, 2, 4) {
            Ok(k) => casts.push(k),
            Err(e) => {
                set_error(format!("cast pipeline: {e}"));
                for k in casts.drain(..) {
                    ctx.destroy_kernel(k);
                }
                for k in kernels.drain(..) {
                    ctx.destroy_kernel(k);
                }
                return std::ptr::null_mut();
            }
        }
    }
    let matmul_f32 = match ctx.kernel(SPV_MATMUL_F32, 3, matmul_ops::PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("matmul_f32 pipeline: {e}"));
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // matmul-f32/v2: the same three buffers and 64-byte block as v1, a
    // different arithmetic contract and tiling (kinds 2 and 3)
    let matmul_f32_v2 = match ctx.kernel(SPV_MATMUL_F32_V2, 3, matmul_ops::PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("matmul_f32_v2 pipeline: {e}"));
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // four storage buffers and the family's 120-byte push block; the push range is
    // checked against the device's limit at creation, so a device that cannot
    // carry it refuses the open by name here
    let pointwise = match ctx.kernel(
        SPV_POINTWISE,
        pointwise_ops::OPERANDS as u32,
        pointwise_ops::PUSH_BYTES as u32,
    ) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("pointwise pipeline: {e}"));
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // reduce-f32/v1 (ABI 7): two storage buffers (the input view, then the
    // partials or the output) and the family's 80-byte push block
    let reduce = match ctx.kernel(SPV_REDUCE, 2, reduce_ops::PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("reduce pipeline: {e}"));
            ctx.destroy_kernel(pointwise);
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // reduce-f32/v2 (2026-09-09): the same two buffers and the kernel's 96-byte
    // push block; a device that cannot build it refuses to open, as for v1
    let reduce_v2 = match ctx.kernel(SPV_REDUCE_V2, 2, reduce_ops::PUSH_BYTES_V2 as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("reduce_v2 pipeline: {e}"));
            ctx.destroy_kernel(reduce);
            ctx.destroy_kernel(pointwise);
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // rowwise-f32/v2 (ABI 8): four storage buffers (primary, upstream, gamma,
    // output) and the kernel's 16-byte push block
    let rowwise = match ctx.kernel(SPV_ROWWISE, 4, ROWWISE_PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("rowwise pipeline: {e}"));
            ctx.destroy_kernel(reduce_v2);
            ctx.destroy_kernel(reduce);
            ctx.destroy_kernel(pointwise);
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // indexed-cross-entropy-f32/v2 (ABI 9): four storage buffers (primary,
    // targets, upstream, output) and the kernel's 20-byte push block
    let loss = match ctx.kernel(SPV_LOSS, 4, LOSS_PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("loss pipeline: {e}"));
            ctx.destroy_kernel(rowwise);
            ctx.destroy_kernel(reduce_v2);
            ctx.destroy_kernel(reduce);
            ctx.destroy_kernel(pointwise);
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    // embedding-f32/v2 (ABI 9): four storage buffers (primary, indices, offsets,
    // output) and the kernel's 16-byte push block
    let embedding = match ctx.kernel(SPV_EMBEDDING, 4, EMBEDDING_PUSH_BYTES as u32) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("embedding pipeline: {e}"));
            ctx.destroy_kernel(loss);
            ctx.destroy_kernel(rowwise);
            ctx.destroy_kernel(reduce_v2);
            ctx.destroy_kernel(reduce);
            ctx.destroy_kernel(pointwise);
            ctx.destroy_kernel(matmul_f32_v2);
            ctx.destroy_kernel(matmul_f32);
            for k in casts.drain(..) {
                ctx.destroy_kernel(k);
            }
            for k in kernels.drain(..) {
                ctx.destroy_kernel(k);
            }
            return std::ptr::null_mut();
        }
    };
    let mut reg = registry();
    if reg.next_device > u32::MAX as u64 {
        // the owner half of a buffer handle is 32 bits; ids are never reused
        set_error("device handle space exhausted in this process");
        return std::ptr::null_mut();
    }
    let id = reg.next_device;
    reg.next_device += 1;
    let dev = Arc::new(AckDevice {
        id,
        inner: Mutex::new(Inner {
            ctx,
            kernels,
            casts,
            matmul_f32: Some(matmul_f32),
            matmul_f32_v2: Some(matmul_f32_v2),
            pointwise: Some(pointwise),
            pointwise_storage: [const { None }; 15],
            reduce: Some(reduce),
            reduce_v2: Some(reduce_v2),
            reduce_storage: [const { None }; 3],
            reduce_v2_storage: [const { None }; 3],
            reduce_v1_only: std::env::var_os("ACK_REDUCE_V1_ONLY").is_some_and(|v| v != "0"),
            rowwise: Some(rowwise),
            loss: Some(loss),
            embedding: Some(embedding),
            vq: None,
            buffers: HashMap::new(),
            next_buffer: 1,
            closed: false,
        }),
    });
    reg.devices.insert(id, dev);
    id as usize as *mut AckDevice
}

/// Close the device. Refused with `ACK_ERR_LIVE_BUFFERS` while any buffer
/// allocated on it has not been freed, in which case the handle stays valid;
/// refused with `ACK_ERR_CLOSED` for a handle that is not open, so a second
/// close of the same handle is an error, never a double free.
///
/// The `closed` flag is what makes this atomic: it is set under the device
/// lock, so of two racing closes exactly one sets it and the other finds it
/// (`ACK_ERR_CLOSED`), and every call that fetched the device beforehand finds
/// it when it takes the lock. The registry lock is held only to look the
/// handle up and, afterwards, to remove it, never while waiting on a device
/// that may be mid-kernel, so a close of one device stalls no other.
///
/// **The last flush point.** After the live-buffer refusal (whose code and
/// timing are unchanged: it still makes no kit call), the deferred batch is
/// submitted and waited for, so recorded work that was already counted as
/// accepted is not discarded silently.
///
/// **A flush failure here does not change this entry's return code.** It is
/// put in `ack_last_error` and `ACK_OK` is still returned. Returning `-2` or
/// `-10` from `ack_close` would be a new circumstance for an existing code at
/// a frozen ABI (9), and the ABI is not moved without the owner saying so;
/// today `ack_close` always succeeds once the buffers are gone, including on a
/// poisoned device, and `tests/poison.rs` pins that. THIS IS AN OPEN QUESTION,
/// NOT A DECISION: silently discarding a failure at the last remaining flush
/// point is the kind of thing this program refuses, and the recommendation is
/// to return it. The owner's call.
#[no_mangle]
pub extern "C" fn ack_close(dev: *mut AckDevice) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    {
        let mut guard = match inner(&device) {
            Ok(g) => g,
            Err(code) => return code,
        };
        let live = guard.buffers.len();
        if live != 0 {
            set_error(format!(
                "close refused: {live} buffer(s) still allocated on this device"
            ));
            return ACK_ERR_LIVE_BUFFERS;
        }
        // the last flush point: recorded work that was accepted must be
        // submitted, not dropped. A failure is reported through
        // `ack_last_error` and does not change this entry's code (see above).
        // Nothing recorded means nothing to submit and, in particular, no
        // message written over the one a poisoned device left behind.
        if guard.ctx.recorded_items() > 0 {
            if let Err(e) = guard.ctx.flush(FlushReason::Close) {
                set_error(format!(
                    "close flushed the deferred batch and it failed: {e}"
                ));
            }
        }
        guard.closed = true;
    }
    registry().devices.remove(&device.id);
    // the Vulkan objects go when the last reference does: here, unless a call
    // that fetched the device a moment ago is still finishing (it then finds
    // `closed` when it takes the lock and refuses)
    drop(device);
    ACK_OK
}

/// Fill `out` (of `size` bytes) with the open device's identity, features and
/// limits. A struct smaller than this library's is refused; a caller must
/// require `size` to equal its own struct's size.
/// # Safety
/// `out` must point to `size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_device_info(
    dev: *const AckDevice,
    out: *mut AckDeviceInfo,
    size: usize,
) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    if size < std::mem::size_of::<AckDeviceInfo>() {
        set_error(format!(
            "AckDeviceInfo of {size} bytes is smaller than this library's {}",
            std::mem::size_of::<AckDeviceInfo>()
        ));
        return ACK_ERR_ABI;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let ctx = &guard.ctx;
    let r = &ctx.report;
    let mut features = 0u64;
    if r.cooperative_matrix {
        features |= ACK_FEATURE_COOPERATIVE_MATRIX;
    }
    if r.bf16_type {
        features |= ACK_FEATURE_BF16_TYPE;
    }
    if r.bf16_dot_product {
        features |= ACK_FEATURE_BF16_DOT_PRODUCT;
    }
    if r.bf16_cooperative_matrix {
        features |= ACK_FEATURE_BF16_COOPERATIVE_MATRIX;
    }
    if ctx.subgroup_size_control {
        features |= ACK_FEATURE_SUBGROUP_SIZE_CONTROL;
    }
    if r.shader_int16 {
        features |= ACK_FEATURE_SHADER_INT16;
    }
    if r.coopmat_shapes.iter().any(|s| {
        s.m == 16
            && s.n == 16
            && s.k == 16
            && s.a_type == "bfloat16"
            && s.b_type == "bfloat16"
            && s.result_type == "float32"
            && s.scope == "subgroup"
    }) {
        features |= ACK_FEATURE_COOPMAT_BF16_16X16X16_SUBGROUP;
    }
    let mut info = AckDeviceInfo {
        size: std::mem::size_of::<AckDeviceInfo>() as u32,
        version: ACK_DEVICE_INFO_VERSION,
        api_version: r.api_version_raw,
        driver_version_raw: r.driver_version_raw,
        driver_id: r.driver_id as u32,
        vendor_id: r.vendor_id,
        device_id: r.device_id,
        device_type: r.device_type,
        device_uuid: r.device_uuid,
        driver_uuid: r.driver_uuid,
        features,
        subgroup_size: r.subgroup_size,
        timestamp_valid_bits: ctx.timestamp_valid_bits,
        timestamp_period_ns: r.timestamp_period_ns,
        max_push_constant_bytes: ctx.max_push_constant_bytes,
        device_local_bytes: r.device_local_bytes,
        max_storage_buffer_bytes: ctx.max_storage_buffer_bytes,
        max_workgroup_invocations: r.max_workgroup_invocations,
        max_workgroup_count: ctx.max_workgroup_count,
        max_workgroup_size: r.max_workgroup_size,
        matmul_tile: TILE,
        matmul_k_multiple: K_MULTIPLE,
        // 0 on a device opened without the BF16 kernels (ABI 4): the record says what exists
        matmul_layouts: if guard.kernels.is_empty() { 0 } else { 0b1111 },
        matmul_operand_dtype: ACK_DTYPE_BF16,
        matmul_result_dtype: ACK_DTYPE_F32,
        cast_modes: 2,
        reserved: 0,
        identity: 0,
        device_name: [0; 256],
        driver_name: [0; 64],
        driver_info: [0; 128],
    };
    info.identity = info_identity(&info);
    fill_c_string(&mut info.device_name, &r.device_name);
    fill_c_string(&mut info.driver_name, &r.driver_name);
    fill_c_string(&mut info.driver_info, &r.driver_info);
    unsafe { std::ptr::write_unaligned(out, info) };
    ACK_OK
}

/// Fill `out` (of `size` bytes) with what the dispatch instrument has counted
/// and timed. A struct smaller than this library's is refused; a caller must
/// require `size` to equal its own struct's size and `version` to equal
/// `ACK_DISPATCH_STATS_VERSION`.
///
/// READ-ONLY AND GENERATION-FREE. It moves no counter, changes no mode and
/// takes no flush, so it is safe to call between passes without disturbing the
/// accounting generation the torch adapter's in-flight guard rests on. It does
/// NOT flush: work recorded and unsubmitted at the moment of the call is
/// counted in `items` and `commands` but not yet in `device_ns`, so a caller
/// that differences two calls around a pass should take the second AFTER a
/// readback has drained the batch. `ack_step_rate.py` and
/// `step_decompose.py` do that already: every pass ends in a loss readback.
/// # Safety
/// `out` must point to `size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_dispatch_stats(
    dev: *const AckDevice,
    out: *mut AckDispatchStats,
    size: usize,
) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    if size < std::mem::size_of::<AckDispatchStats>() {
        set_error(format!(
            "AckDispatchStats of {size} bytes is smaller than this library's {}",
            std::mem::size_of::<AckDispatchStats>()
        ));
        return ACK_ERR_ABI;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let s = guard.ctx.dispatch_stats();
    let stats = AckDispatchStats {
        size: std::mem::size_of::<AckDispatchStats>() as u32,
        version: ACK_DISPATCH_STATS_VERSION,
        items: s.items,
        commands: s.commands,
        barriers: s.barriers,
        flushes: s.flushes,
        timed_flushes: s.timed_flushes,
        timed_items: s.timed_items,
        device_ns: s.device_ns,
        min_item_ns: s.min_item_ns,
        max_item_ns: s.max_item_ns,
        unresolved_flushes: s.unresolved_flushes,
        timestamps_enabled: u32::from(s.timestamps_enabled),
        reserved: 0,
    };
    unsafe { std::ptr::write_unaligned(out, stats) };
    ACK_OK
}

/// Vulkan objects the device's context has created and not destroyed
/// (ACK-L0-05): counted by the runtime's create and destroy wrappers. A
/// poisoned device keeps its leaked objects counted here; the kernels built
/// at open are counted (four objects each) and go at close.
/// # Safety
/// `out` must point to a writable i64.
#[no_mangle]
pub unsafe extern "C" fn ack_live_objects(dev: *const AckDevice, out: *mut i64) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    unsafe { *out = guard.ctx.live_objects() };
    ACK_OK
}

/// Objects left allocated on purpose because the device is poisoned
/// (ACK-L0-03); 0 on a healthy or lost device.
/// # Safety
/// `out` must point to a writable u64.
#[no_mangle]
pub unsafe extern "C" fn ack_leaked_on_poison(dev: *const AckDevice, out: *mut u64) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    unsafe { *out = guard.ctx.leaked_on_poison() };
    ACK_OK
}

/// Copy the device name (NUL-terminated, truncated to `len`) into `out`.
/// # Safety
/// `out` must point to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_device_name(
    dev: *const AckDevice,
    out: *mut c_char,
    len: usize,
) -> i32 {
    if out.is_null() || len == 0 {
        return ACK_ERR_NULL;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let name = guard.ctx.report.device_name.as_bytes();
    let n = name.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(name.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
    }
    ACK_OK
}

/// Copy the last error message for this thread into `out`.
/// # Safety
/// `out` must point to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_last_error(out: *mut c_char, len: usize) -> i32 {
    if out.is_null() || len == 0 {
        return ACK_ERR_NULL;
    }
    LAST_ERROR.with(|e| {
        let message = e.borrow();
        let bytes = message.as_bytes();
        let n = bytes.len().min(len - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
            *out.add(n) = 0;
        }
    });
    ACK_OK
}

/// A device-local buffer of `bytes`; null on failure with `ack_last_error` set.
/// The handle names this device and this buffer; it is refused, not
/// dereferenced, on any other device or after `ack_buffer_free`.
#[no_mangle]
pub extern "C" fn ack_buffer_alloc(dev: *const AckDevice, bytes: u64) -> *mut Buffer {
    if bytes == 0 {
        set_error("zero-byte buffer");
        return std::ptr::null_mut();
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(_) => return std::ptr::null_mut(),
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(_) => return std::ptr::null_mut(),
    };
    if guard.next_buffer == u32::MAX {
        set_error("buffer handle space exhausted on this device; close and reopen");
        return std::ptr::null_mut();
    }
    match guard.ctx.buffer(bytes) {
        Ok(b) => {
            let seq = guard.next_buffer;
            guard.next_buffer += 1;
            guard.buffers.insert(seq, b);
            buffer_handle(device.id, seq) as usize as *mut Buffer
        }
        Err(e) => {
            set_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a buffer. `ACK_ERR_FOREIGN` for another device's buffer (neither
/// device is touched), `ACK_ERR_FREED` for one already freed.
///
/// **`ACK_OK` no longer means the Vulkan objects are gone.** Since deferred
/// submission the handle is consumed and the id leaves the live set at once
/// (so a dispatch on it is still refused `ACK_ERR_FREED` immediately), but if
/// dispatches are recorded and unsubmitted the buffer and its memory are
/// RETIRED and released by the next flush -- recorded commands may still name
/// them. `ack_live_objects` counts a retired pair until then. Making the free
/// itself a flush point was rejected: the torch adapter frees a kit buffer on
/// every tensor death, so it would flush hundreds of times a step and remove
/// the whole gain; a retirement budget bounds the held memory instead.
///
/// **AND `ACK_OK` NO LONGER MEANS THE OBJECTS WILL BE GONE AFTER THE FLUSH
/// EITHER.** `Context` keeps a free list of released `(VkBuffer,
/// VkDeviceMemory)` pairs, so what the flush does with a retired pair is hand
/// it to that pool rather than destroy it, and `ack_live_objects` goes on
/// counting it -- correctly: it is live, and the next allocation of the same
/// size will be served from it without a Vulkan call. That is a change in what
/// the count MEANS for a freed buffer, so it is stated here rather than left
/// to be discovered: an object being live no longer implies a caller holds it.
/// `ACK_BUFFER_POOL=0` restores the pre-pool behaviour exactly.
///
/// **A free CAN now fail for the batch rather than for the buffer.** Crossing
/// `ACK_MAX_RETIRED_BYTES` makes this entry flush, and that flush's failure is
/// returned here rather than dropped: the handle is consumed either way, and
/// the alternative is a device error that nothing in the stack ever reports.
/// So a free can return `-2` or `-10` naming a BATCH whose dispatches some
/// earlier call recorded. No code and no signature moved -- these are the two
/// codes this entry already returned -- but the CIRCUMSTANCES widened, and
/// that is flagged rather than hidden (see `ACK_ABI_VERSION`).
///
/// `free_refused` in the torch adapter is unchanged in meaning: it moves only
/// when this entry returns non-zero. What changes is that a retired buffer
/// whose flush later poisons the device is leaked and counted THEN, after this
/// call already returned `ACK_OK`.
#[no_mangle]
pub extern "C" fn ack_buffer_free(dev: *const AckDevice, buf: *mut Buffer) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = buffer_of(&device, &guard, buf) {
        return code;
    }
    let (_, seq) = split_buffer_handle(buf as usize as u64);
    if let Some(b) = guard.buffers.remove(&seq) {
        if let Err(e) = guard.ctx.release_buffer(b) {
            // the handle is consumed either way. `Poisoned` here is this
            // buffer's own story -- it was leaked, not freed; anything else is
            // the retirement flush's, and belongs to the batch, not to this
            // buffer, so it must not be reported as a leak of it.
            let what = match e {
                AckError::Poisoned(_) => "buffer leaked, not freed",
                _ => "the buffer was retired and the flush its retirement triggered failed",
            };
            set_error(format!("{what}: {e}"));
            return code_for(&e);
        }
    }
    ACK_OK
}

/// Test-only (`fault-injection` feature): arm a substituted result for this
/// device's next fence wait (`kind` 0) or device idle (`kind` 1), applied after
/// the real call has run. `result` is the raw `VkResult` value. Absent from a
/// build without the feature, so a binding cannot reach it by accident.
#[cfg(feature = "fault-injection")]
#[no_mangle]
pub extern "C" fn ack_inject_fault(dev: *const AckDevice, kind: i32, result: i32) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let r = ash::vk::Result::from_raw(result);
    let fault = match kind {
        0 => crate::Fault::WaitFails(r),
        1 => crate::Fault::IdleFails(r),
        _ => {
            set_error(format!(
                "ack_inject_fault: kind {kind} is not 0 (wait) or 1 (idle)"
            ));
            return ACK_ERR_SHAPE;
        }
    };
    guard.ctx.inject_fault(fault);
    ACK_OK
}

/// Copy `len` host bytes into the buffer (must equal the buffer size).
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_upload(
    dev: *const AckDevice,
    buf: *const Buffer,
    data: *const u8,
    len: usize,
) -> i32 {
    if data.is_null() {
        return ACK_ERR_NULL;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let buffer = match buffer_of(&device, &guard, buf) {
        Ok(b) => b,
        Err(code) => return code,
    };
    if len as u64 != buffer.bytes {
        set_error(format!(
            "upload of {len} bytes into a {}-byte buffer",
            buffer.bytes
        ));
        return ACK_ERR_SIZE;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    match guard.ctx.upload(buffer, slice) {
        Ok(()) => ACK_OK,
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

/// Copy the buffer into `len` host bytes (must equal the buffer size).
/// # Safety
/// `data` must point to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_download(
    dev: *const AckDevice,
    buf: *const Buffer,
    data: *mut u8,
    len: usize,
) -> i32 {
    if data.is_null() {
        return ACK_ERR_NULL;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let buffer = match buffer_of(&device, &guard, buf) {
        Ok(b) => b,
        Err(code) => return code,
    };
    if len as u64 != buffer.bytes {
        set_error(format!(
            "download of {len} bytes from a {}-byte buffer",
            buffer.bytes
        ));
        return ACK_ERR_SIZE;
    }
    match guard.ctx.download(buffer) {
        Ok(bytes) => {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, len) };
            ACK_OK
        }
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

/// Elementwise cast of `n` elements: `mode` 0 is f32 -> bf16 (round to nearest
/// even), 1 is bf16 -> f32. `src` and `dst` are device buffers sized for `n`
/// elements of their dtype; anything smaller is refused.
#[no_mangle]
pub extern "C" fn ack_cast(
    dev: *const AckDevice,
    src: *const Buffer,
    dst: *const Buffer,
    n: u64,
    mode: i32,
) -> i32 {
    if n == 0 || n > u32::MAX as u64 || !(mode == 0 || mode == 1) {
        set_error(format!("cast: bad n={n} or mode={mode}"));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let src = match buffer_of(&device, &guard, src) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let dst = match buffer_of(&device, &guard, dst) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let (src_bytes, dst_bytes) = if mode == 0 {
        (n * 4, n * 2)
    } else {
        (n * 2, n * 4)
    };
    if src.bytes < src_bytes || dst.bytes < dst_bytes {
        set_error(format!("cast: buffers too small for n={n} mode={mode}"));
        return ACK_ERR_SIZE;
    }
    let kernel = &guard.casts[mode as usize];
    if let Err(e) = guard.ctx.bind(kernel, &[src, dst]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let push = (n as u32).to_le_bytes();
    let groups = [(n as u32).div_ceil(CAST_ELEMENTS_PER_GROUP), 1, 1];
    // the declared access set: slot order, reads then the write (the cast
    // entry has no unused slot). It is checked against the binding and is
    // otherwise inert in phase 1 -- R1 barriers every adjacent pair whatever
    // they touch (see `batch::check_declared_access`).
    let access = [Access::read(src.id()), Access::write(dst.id())];
    recorded(guard.ctx.dispatch(kernel, &push, groups, 1, &access))
}

/// `C[M,N] = op(A) . op(B)`, BF16 operands, f32 result.
///
/// The dispatch is RECORDED and runs
/// at the next flush, so `ms_out` is NaN: no duration exists for this call to
/// report (increment 5, deferred submission).
/// # Safety
/// `ms_out` must be null or point to a writable f64.
#[no_mangle]
pub unsafe extern "C" fn ack_matmul_bf16(
    dev: *const AckDevice,
    a: *const Buffer,
    b: *const Buffer,
    c: *const Buffer,
    m: u32,
    n: u32,
    k: u32,
    a_t: i32,
    b_t: i32,
    ms_out: *mut f64,
) -> i32 {
    if m == 0 || n == 0 || k == 0 {
        set_error(format!(
            "shape {m}x{n}x{k} has a zero dimension: the dispatch grid would be empty"
        ));
        return ACK_ERR_SHAPE;
    }
    // Off-tile shapes run the -DEDGE=1 variants, which bounds-check every
    // staged element and predicate the store. Aligned shapes keep running the
    // aligned modules: same pipeline, same bytes, same result as before.
    let edge = m % TILE != 0 || n % TILE != 0 || k % K_MULTIPLE != 0;
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let a = match buffer_of(&device, &guard, a) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let b = match buffer_of(&device, &guard, b) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let c = match buffer_of(&device, &guard, c) {
        Ok(b) => b,
        Err(code) => return code,
    };
    // The exact dense sizes, unchanged. The edge variants read A and B through
    // a `uint` view and so touch the whole word holding the last element, but
    // that is covered by `Context::buffer` allocating every buffer to a whole
    // word rather than by demanding a rounded size here: requiring it here
    // would refuse a legitimate odd-extent product whose operands the caller
    // sized exactly, which is what the torch adapter does.
    let need_a = m as u64 * k as u64 * 2;
    let need_b = k as u64 * n as u64 * 2;
    let need_c = m as u64 * n as u64 * 4;
    if a.bytes < need_a || b.bytes < need_b || c.bytes < need_c {
        set_error(format!(
            "buffers too small for {m}x{n}x{k}: A {} (need {need_a}), B {} (need {need_b}), C {} (need {need_c})",
            a.bytes, b.bytes, c.bytes
        ));
        return ACK_ERR_SIZE;
    }
    if guard.kernels.is_empty() {
        set_error(
            "BF16 cooperative-matrix matmul is not available on this device (no BF16 cooperative matrices or no subgroup size control); ack_matmul_f32 is",
        );
        return ACK_ERR_UNSUPPORTED_OP;
    }
    let index = (edge as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize);
    let kernel = &guard.kernels[index];
    if let Err(e) = guard.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // a single product: z = 1 with every z word zero, the path it always took
    let push = coopmat_push(m, n, k, 0, 0, 0, 0, 0, 0, 0);
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    // div_ceil, not `/`: a truncating grid never launches the partial edge tile,
    // so those rows and columns of C are left UNWRITTEN while the call still
    // returns ACK_OK and every counter reports success. Identical to `/` for
    // aligned shapes.
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &push,
        [n.div_ceil(TILE), m.div_ceil(TILE), 1],
        1,
        &access,
    ));
    if rc == ACK_OK {
        // recorded, not run: no duration exists for this call to report
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// `ack_matmul_bf16` with a z grid (ABI 10, matmul-bf16 schema 2).
///
/// `zmode` 0: `z` independent products, the b-th reading A at `b * za`
/// elements, B at `b * zb` and writing C at `b * zc` (a torch `bmm` with
/// dense per-batch blocks is `za = m*k`, `zb = k*n`, `zc = m*n`). `zmode` 1:
/// `z` K-splits of ONE product, the s-th owning `k` in
/// `[s*kz, min(k, (s+1)*kz))` and writing its PARTIAL product at `s * zc`;
/// the caller sums the partials over the splits (the reduce family does that
/// in one dispatch). `za` and `zb` must be zero in mode 1. Every product of a
/// call has the same M, N, K and layouts; the modules are chosen as for a
/// single product, except that an aligned module also needs `za` and `zb` to
/// be multiples of eight elements (its operands are read as uvec4) and `kz`
/// a multiple of the K tile, so a call that breaks either takes the edge
/// modules. `z` is at most 65,535 (the Vulkan guarantee for a z grid).
/// `lda` and `ldb` are the STORED row pitch of A and of B in elements (A
/// row-major: its M rows of K; A transposed: its K stored rows of M; B
/// row-major: K rows of N; B transposed: N rows of K), or 0 for the logical
/// width; a pitch below the width is refused, and an aligned module needs
/// both pitches to be multiples of eight (else the edge modules). A
/// head-interleaved [batch, rows, cols] view is read in place through the
/// z stride and the pitch, with no staging copy.
/// Refusals are by code with the reason in `ack_last_error`, and nothing is
/// dispatched on one. A call with `z = 1` and every z word zero is
/// `ack_matmul_bf16`.
///
/// The dispatch is RECORDED and runs at the next flush, so `ms_out` is NaN.
/// # Safety
/// `ms_out` must be null or point to a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_bf16_z(
    dev: *const AckDevice,
    a: *const Buffer,
    b: *const Buffer,
    c: *const Buffer,
    m: u32,
    n: u32,
    k: u32,
    a_t: i32,
    b_t: i32,
    z: u32,
    zmode: u32,
    za: u32,
    zb: u32,
    zc: u32,
    kz: u32,
    lda: u32,
    ldb: u32,
    ms_out: *mut f64,
) -> i32 {
    if m == 0 || n == 0 || k == 0 {
        set_error(format!(
            "shape {m}x{n}x{k} has a zero dimension: the dispatch grid would be empty"
        ));
        return ACK_ERR_SHAPE;
    }
    if z == 0 || z > MAX_GRID_Z {
        set_error(format!("z {z} is outside 1..={MAX_GRID_Z}"));
        return ACK_ERR_SHAPE;
    }
    if zmode > 1 {
        set_error(format!("zmode {zmode} is not 0 (batch) or 1 (K-split)"));
        return ACK_ERR_SHAPE;
    }
    let mn = u64::from(m) * u64::from(n);
    // stored shapes: A is (m, k) or, transposed, (k, m); B is (k, n) or (n, k)
    let (a_rows, a_cols) = if a_t != 0 { (k, m) } else { (m, k) };
    let (b_rows, b_cols) = if b_t != 0 { (n, k) } else { (k, n) };
    let lda = if lda == 0 { a_cols } else { lda };
    let ldb = if ldb == 0 { b_cols } else { ldb };
    if lda < a_cols || ldb < b_cols {
        set_error(format!(
            "pitch {lda}/{ldb} below the stored width {a_cols}/{b_cols}"
        ));
        return ACK_ERR_SHAPE;
    }
    if zmode == 1 {
        if za != 0 || zb != 0 {
            set_error("K-split mode takes no operand z strides (za and zb must be 0)");
            return ACK_ERR_SHAPE;
        }
        if kz == 0 || u64::from(kz) * u64::from(z) < u64::from(k) {
            set_error(format!("{z} splits of {kz} do not cover k = {k}"));
            return ACK_ERR_SHAPE;
        }
        if u64::from(zc) < mn {
            set_error(format!(
                "partials {zc} elements apart overlap an {m}x{n} product"
            ));
            return ACK_ERR_SHAPE;
        }
    } else if z > 1 && (za < a_cols || zb < b_cols || u64::from(zc) < mn) {
        // an operand's z stride below its stored row width would make two
        // products read overlapping rows; the output's below one product's
        // block would make them write over each other. A z stride below the
        // whole slab is legitimate: a head-interleaved view has z stride 64
        // under a row pitch of 512.
        set_error(format!(
            "z strides {za}/{zb}/{zc} below the row widths {a_cols}/{b_cols} or the {m}x{n} output block"
        ));
        return ACK_ERR_SHAPE;
    }
    let edge = m % TILE != 0
        || n % TILE != 0
        || k % K_MULTIPLE != 0
        || lda % Z_STRIDE_MULTIPLE != 0
        || ldb % Z_STRIDE_MULTIPLE != 0
        || (zmode == 0 && (za % Z_STRIDE_MULTIPLE != 0 || zb % Z_STRIDE_MULTIPLE != 0))
        || (zmode == 1 && kz % K_MULTIPLE != 0);
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let a = match buffer_of(&device, &guard, a) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let b = match buffer_of(&device, &guard, b) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let c = match buffer_of(&device, &guard, c) {
        Ok(b) => b,
        Err(code) => return code,
    };
    if c.id() == a.id() || c.id() == b.id() {
        set_error("the output buffer must be distinct from both inputs");
        return ACK_ERR_SHAPE;
    }
    let last = u64::from(z - 1);
    let a_span = u64::from(a_rows - 1) * u64::from(lda) + u64::from(a_cols);
    let b_span = u64::from(b_rows - 1) * u64::from(ldb) + u64::from(b_cols);
    let (need_a, need_b, need_c) = if zmode == 1 {
        (a_span * 2, b_span * 2, (last * u64::from(zc) + mn) * 4)
    } else {
        (
            (last * u64::from(za) + a_span) * 2,
            (last * u64::from(zb) + b_span) * 2,
            (last * u64::from(zc) + mn) * 4,
        )
    };
    if a.bytes < need_a || b.bytes < need_b || c.bytes < need_c {
        set_error(format!(
            "buffers too small for {z} x {m}x{n}x{k} (zmode {zmode}): A {} (need {need_a}), B {} (need {need_b}), C {} (need {need_c})",
            a.bytes, b.bytes, c.bytes
        ));
        return ACK_ERR_SIZE;
    }
    if guard.kernels.is_empty() {
        set_error(
            "BF16 cooperative-matrix matmul is not available on this device (no BF16 cooperative matrices or no subgroup size control); ack_matmul_f32 is",
        );
        return ACK_ERR_UNSUPPORTED_OP;
    }
    let index = (edge as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize);
    let kernel = &guard.kernels[index];
    if let Err(e) = guard.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let push = coopmat_push(m, n, k, zmode, za, zb, zc, kz, lda, ldb);
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &push,
        [n.div_ceil(TILE), m.div_ceil(TILE), z],
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// `C = A . B` in FP32 on the device. Kinds 0 (one batch) and 1 (batched)
/// run the near-exact compensated matmul-f32/v1 kernel; kinds 2 and 3 (schema
/// 2, 2026-09-09) run the same products on matmul-f32/v2, plain fp32
/// accumulation on a 64 x 64 tile, through the SAME block and buffers, so a
/// plan v1 refuses v2 refuses identically and only the arithmetic and the
/// dispatch grid differ. `plan` is the family's 64-byte little-endian push block: `batch, m, n, k`, then `offset,
/// batch_stride, row_stride, col_stride` in elements for A, B and C. The block
/// is re-validated here by `matmul_ops::MatmulPlan` against the buffers' real
/// capacities (no view may address past its buffer) and the output must be a
/// buffer distinct from both inputs. Refusals are by name before any dispatch:
/// `ACK_ERR_NULL` for a null block, `ACK_ERR_SIZE` for a block of another
/// length or a view past its buffer, `ACK_ERR_SHAPE` for everything the plan
/// refuses (a batch above 4096, `m` or `n` above 65,536, a reduction length
/// `k` above 65,536, more than 268,435,456 addressed elements per view, zero
/// strides, mismatched dimensions, an overlapping output, an unknown kind).
///
/// The shader's own geometry check cannot fire once this validation has passed
/// (every view is inside its buffer); if it did, the dispatch would write
/// nothing and this entry would still return `ACK_OK`. That hazard is real and
/// is now guarded rather than merely described: `tests/matmul_plan.rs` asserts
/// that the shader's capacity literals, and the constants in the compiled
/// `.spv`, are the same numbers as `matmul_ops`'s, and `tests/matmul_f32_abi.rs`
/// dispatches a poisoned output at each bound on a real device and requires
/// that no sentinel element survives. Fixed-fixture numerical evidence only;
/// see MATMUL_KERNELS.md.
/// # Safety
/// `plan` must point to `plan_len` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_matmul_f32(
    dev: *const AckDevice,
    a: *const Buffer,
    b: *const Buffer,
    c: *const Buffer,
    kind: u32,
    plan: *const u8,
    plan_len: usize,
    ms_out: *mut f64,
) -> i32 {
    if plan.is_null() {
        return ACK_ERR_NULL;
    }
    if plan_len != matmul_ops::PUSH_BYTES {
        set_error(format!(
            "matmul_f32: the plan block is {plan_len} bytes, not {}",
            matmul_ops::PUSH_BYTES
        ));
        return ACK_ERR_SIZE;
    }
    let kind = match matmul_ops::MatmulKind::try_from(kind) {
        Ok(k) => k,
        Err(e) => {
            set_error(format!("matmul_f32: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let bytes = unsafe { std::slice::from_raw_parts(plan, plan_len) };
    let mut words = [0u32; matmul_ops::PUSH_BYTES / 4];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let a = match buffer_of(&device, &guard, a) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let b = match buffer_of(&device, &guard, b) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let c = match buffer_of(&device, &guard, c) {
        Ok(x) => x,
        Err(code) => return code,
    };
    if c.id() == a.id() || c.id() == b.id() {
        set_error("matmul_f32: the output buffer must be distinct from both inputs");
        return ACK_ERR_SHAPE;
    }
    let (batch, m, n, k_dim) = (words[0], words[1], words[2], words[3]);
    let view = |rows: u32, cols: u32, w: &[u32], capacity_bytes: u64| {
        matmul_ops::MatrixView::new(
            batch,
            rows,
            cols,
            u64::from(w[0]),
            u64::from(w[1]),
            u64::from(w[2]),
            u64::from(w[3]),
            capacity_bytes / 4,
        )
    };
    let plan = view(m, k_dim, &words[4..8], a.bytes)
        .and_then(|va| view(k_dim, n, &words[8..12], b.bytes).map(|vb| (va, vb)))
        .and_then(|(va, vb)| view(m, n, &words[12..16], c.bytes).map(|vc| (va, vb, vc)))
        .and_then(|(va, vb, vc)| matmul_ops::MatmulPlan::new(kind, va, vb, vc));
    let plan = match plan {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("matmul_f32 plan refused: {e}"));
            return match e {
                matmul_ops::MatmulPlanError::CapacityTooSmall => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    // the kind names the kernel: v2 for the plain kinds, v1 otherwise; the
    // plan already cut its grid to that kernel's tile
    let slot = if kind.is_plain() {
        guard.matmul_f32_v2.as_ref()
    } else {
        guard.matmul_f32.as_ref()
    };
    let kernel = match slot {
        Some(k) => k,
        None => {
            set_error("matmul_f32: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// `out = f(x, y, z)` over one row-major shape of at most four dimensions in
/// FP32 on the device (pointwise-f32/v1, schema 2): the 24 operations of
/// `pointwise_ops::PointwiseOp` (schema 1's `x + a*y`, `x - a*y`, `x*y`,
/// `x/y`, fill(a), copy(x); schema 2's reciprocal, abs, sqrt, rsqrt, neg,
/// pow with a tensor or scalar exponent, pow with a scalar base, clamp, silu,
/// sigmoid, sigmoid_backward, addcmul, addcdiv, lerp, cos, sin, iota and
/// triu, each documented with its operand slots and scalar words on the
/// variant and in the shader header), every operand through its own element
/// offset and per-dimension element strides with a broadcast as stride 0,
/// and `y` optionally the scalar carried in the block (flags bit 0). `plan`
/// is the family's 120-byte little-endian push block, the layout
/// `pointwise_ops::PointwisePlan::push_constants` writes. The block is
/// re-validated here by `PointwisePlan::from_push_constants` against the
/// buffers' real capacities and identities, so an input may share the
/// output's buffer only through a view identical to the output's (in place);
/// `z` (binding 2) is read by addcmul and addcdiv only, but bound and
/// validated for every operation, so bind it to `x` when unused. Refusals
/// are by name before any dispatch: `ACK_ERR_NULL` for a null block,
/// `ACK_ERR_SIZE` for a block of another length or for a view past its
/// buffer or the 2^31 address limit (`pointwise-address-overflow`),
/// `ACK_ERR_SHAPE` for everything else the plan refuses (an unknown
/// operation, decoded before any device lookup; more than four dimensions;
/// triu over fewer than two (`pointwise-triu-needs-two-dimensions`); zero
/// elements; an output overlapping itself or an input through another view;
/// a reserved flag; an element count that is not the shape's) and for an
/// element count needing more workgroups than the device's first axis
/// holds. The shader's own extent check cannot fire once this validation has
/// passed (every view is inside its buffer); if it did, the dispatch would
/// write nothing and this entry would still return `ACK_OK` (UNMEASURED: no
/// input reaches it). The dispatch is RECORDED and runs
/// at the next flush, so `ms_out` is NaN: no duration exists for this call to
/// report (increment 5, deferred submission).
/// # Safety
/// `plan` must point to `plan_len` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_pointwise(
    dev: *const AckDevice,
    x: *const Buffer,
    y: *const Buffer,
    z: *const Buffer,
    out: *const Buffer,
    plan: *const u8,
    plan_len: usize,
    ms_out: *mut f64,
) -> i32 {
    if plan.is_null() {
        return ACK_ERR_NULL;
    }
    if plan_len != pointwise_ops::PUSH_BYTES {
        set_error(format!(
            "pointwise: the plan block is {plan_len} bytes, not {}",
            pointwise_ops::PUSH_BYTES
        ));
        return ACK_ERR_SIZE;
    }
    let mut block = [0u8; pointwise_ops::PUSH_BYTES];
    block.copy_from_slice(unsafe { std::slice::from_raw_parts(plan, plan_len) });
    // the operation word is decoded before any device lookup, as the matmul's kind is
    let op_word = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    if let Err(e) = pointwise_ops::PointwiseOp::try_from(op_word) {
        set_error(format!("pointwise: plan refused: {e}"));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    // mutable because a bf16 storage module is built on the first call that
    // asks for one; see the module selection below
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // The storage module this block asks for, read from its flags before any
    // buffer is resolved: 0 is the f32 module every device builds at open, and
    // 1..3 are the bf16 ones, built HERE on the first call that asks for one,
    // so a device that never sees a bf16 operand never builds them. It happens
    // before the buffers are borrowed because the guard cannot be borrowed
    // mutably while they are alive.
    let module = pointwise_ops::storage_module(u32::from_le_bytes([
        block[12], block[13], block[14], block[15],
    ]));
    if module != 0 && guard.pointwise_storage[module - 1].is_none() {
        let state: &mut Inner = &mut guard;
        let built = state.ctx.kernel(
            SPV_POINTWISE_STORAGE[module - 1],
            pointwise_ops::OPERANDS as u32,
            pointwise_ops::PUSH_BYTES as u32,
        );
        match built {
            Ok(k) => state.pointwise_storage[module - 1] = Some(k),
            Err(e) => {
                set_error(format!("pointwise storage pipeline {module}: {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let x = match buffer_of(&device, &guard, x) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let y = match buffer_of(&device, &guard, y) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let z = match buffer_of(&device, &guard, z) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    // the runtime's buffer id is the allocation's identity (one id per live
    // buffer, so two handles naming one buffer agree), and the capacity is the
    // real byte length in ELEMENTS, which is two bytes under bf16 storage and
    // four otherwise; the inputs and the output are asked separately because
    // f32 in with bf16 out is the strided cast this family now carries
    let flags_word = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
    // Every slot is measured by ITS OWN storage bit. A slot the operation does
    // not read is bound to whatever the caller had to hand -- the output for
    // `zero_`, the source for a copy -- so there is nothing to infer it from,
    // and a guess that was too generous would weaken its capacity check.
    let storage = |b: &Buffer, index: usize| {
        pointwise_ops::Storage::new(
            b.id(),
            b.bytes / pointwise_ops::slot_element_bytes(flags_word, index),
        )
    };
    let plan = match pointwise_ops::PointwisePlan::from_push_constants(
        &block,
        [storage(x, 0), storage(y, 1), storage(z, 2), storage(out, 3)],
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("pointwise: plan refused: {e}"));
            return match e {
                pointwise_ops::PointwisePlanError::AddressOverflow => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    // Defensive: ack_open fails when the f32 pipeline cannot be built, so an
    // open device always carries it; the branch names the state rather than
    // trusting it.
    // the plan's own reading of the same bits; it refused the block if they
    // were not ones this version implements, so the two agree by construction
    let selected = plan.storage_module();
    let slot = if selected == 0 {
        guard.pointwise.as_ref()
    } else {
        guard.pointwise_storage[selected - 1].as_ref()
    };
    let kernel = match slot {
        Some(k) => k,
        None => {
            set_error("pointwise: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[x, y, z, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // `y` and `z` are bound to live buffers even for an operation that reads
    // neither (the plan's own law); a slot the operation does not use is
    // declared Read, which is conservative in both directions R2 could ever go
    let access = [
        Access::read(x.id()),
        Access::read(y.id()),
        Access::read(z.id()),
        Access::write(out.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// The request a caller of `ack_reduce` passes: the kernel's own 80-byte push
/// layout (`reduce_ops::push_bytes`) with the two words the plan derives left
/// zero. Decoded before any device lookup, every defect by name.
struct ReduceRequest {
    op: ReduceOp,
    finish: Finish,
    ndim: usize,
    reduced_mask: u32,
    shape: [u32; reduce_ops::MAX_RANK],
    stride: [u32; reduce_ops::MAX_RANK],
    offset: u32,
    scale: f32,
}

impl ReduceRequest {
    /// Word 17 (`splits`) and word 18 (`chunk`) are the plan's to derive; a
    /// caller that sets them is refused rather than second-guessed, and the
    /// padding beyond `ndim` must be the view's own (extent 1, stride 0).
    fn decode(block: &[u8; reduce_ops::PUSH_BYTES]) -> Result<Self, &'static str> {
        let word = |i: usize| {
            u32::from_le_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ])
        };
        let op = match word(0) {
            0 => ReduceOp::Sum,
            1 => ReduceOp::Amax,
            2 => ReduceOp::SumOfSquares,
            _ => return Err("reduce-operation-out-of-range"),
        };
        let finish = match word(1) {
            0 => Finish::None,
            1 => Finish::Scale,
            2 => Finish::Sqrt,
            _ => return Err("reduce-finish-out-of-range"),
        };
        let ndim = word(2) as usize;
        if ndim == 0 || ndim > reduce_ops::MAX_RANK {
            return Err("reduce-rank-out-of-range");
        }
        // The kernel walks the reduced index space with a 32-bit `uint`, while
        // `ReducePlan` carries the count as u64 and bounds only the splits, the
        // kept count and the workgroup total. A view whose reduced dimensions
        // are BROADCAST (stride 0) keeps `last_address` small while multiplying
        // the reduced extents without limit, so a plan the module admits can
        // wrap the kernel's counter and return a plausible wrong number. Bound
        // it here, from the request alone, before any device lookup.
        let mut reduced: u64 = 1;
        let mask = word(3);
        for d in 0..ndim {
            if mask & (1 << d) != 0 {
                reduced = reduced.saturating_mul(word(4 + d) as u64);
            }
        }
        if reduced > u32::MAX as u64 {
            return Err("reduce-reduced-count-above-u32");
        }
        if word(17) != 0 || word(18) != 0 {
            return Err("reduce-reserved-word-nonzero");
        }
        let mut shape = [1u32; reduce_ops::MAX_RANK];
        let mut stride = [0u32; reduce_ops::MAX_RANK];
        for d in 0..reduce_ops::MAX_RANK {
            shape[d] = word(4 + d);
            stride[d] = word(10 + d);
            if d >= ndim && (shape[d] != 1 || stride[d] != 0) {
                return Err("reduce-padding-out-of-range");
            }
        }
        Ok(Self {
            op,
            finish,
            ndim,
            reduced_mask: word(3),
            shape,
            stride,
            offset: word(16),
            scale: f32::from_bits(word(19)),
        })
    }
}

fn reduce_plan_code(err: ReducePlanError) -> i32 {
    match err {
        ReducePlanError::AddressOutOfRange | ReducePlanError::CapacityTooSmall => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// `out[kept] = reduce(x)` over the reduced dimensions of a strided FP32 view
/// on the device (reduce-f32/v1, schema 1): the operations of
/// `reduce_ops::ReduceOp` (sum, NaN-propagating amax, sum of squares), finished
/// by nothing, a scale or a square root (`reduce_ops::Finish`), in the fixed
/// combination order the plan module defines, so `aten::sum.dim_IntList`,
/// `mean.dim`, `amax` and `linalg_vector_norm` (ord 2) over any subset of at
/// most six dimensions. `plan` is the family's 80-byte little-endian request
/// block in the kernel's own push layout: op, finish, ndim, reduced mask, six
/// extents, six element strides (0 broadcasts), offset, then `splits` and
/// `chunk` which the caller MUST leave 0 (the plan derives them), then the
/// scale bits (`Finish::Scale` multiplies by it; a mean passes the reciprocal
/// of the reduced count). The plan is re-derived here by
/// `ReducePlan::new` against `x`'s real capacity. The output is dense from
/// element 0 of `out`, one value per kept index with the kept dimensions in
/// their original order. A plan whose reduced count exceeds
/// `reduce_ops::CHUNK` (65,536) runs two passes through `partials`, a scratch
/// buffer the caller sizes to at least `kept * splits` elements
/// (`splits = ceil(reduced / 65,536)`); `partials` may be null for a one-pass
/// plan. A reduction is never in place: `out` and `partials` must be buffers
/// other than `x`'s and each other's. Refusals are by name before any
/// dispatch: `ACK_ERR_NULL` for a null block or a two-pass plan without a
/// partials buffer; `ACK_ERR_SIZE` for a block of another length, a view past
/// the 2^31 address limit or past `x` (`reduce-capacity-too-small`), an output
/// (`reduce-output-too-small`) or partials buffer (`reduce-partials-too-small`)
/// with too few elements; `ACK_ERR_SHAPE` for an unknown operation or finish
/// word or a nonzero reserved word (decoded before any device lookup), a rank
/// outside 1..=6, a zero extent, a reduced mask naming no dimension or one
/// beyond the rank, a non-finite scale, too many workgroups, and the aliasing
/// above (`reduce-output-aliases-input`, `reduce-partials-alias`).
///
/// The dispatches are RECORDED and run at the next flush, so `ms_out` is NaN:
/// no duration exists for this call to report (increment 5, deferred
/// submission). BOTH PASSES GO INTO ONE BATCH. The second pass reads what the
/// first wrote, ordered by the barrier the recorder puts between adjacent
/// items, and each pass takes its OWN descriptor set from the kernel's arena
/// -- which is what makes rebinding between them safe now that the first pass
/// is no longer submitted and waited for before the second is recorded.
/// # Safety
/// `plan` must point to `plan_len` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_reduce(
    dev: *const AckDevice,
    x: *const Buffer,
    partials: *const Buffer,
    out: *const Buffer,
    plan: *const u8,
    plan_len: usize,
    ms_out: *mut f64,
) -> i32 {
    if plan.is_null() {
        return ACK_ERR_NULL;
    }
    if plan_len != reduce_ops::PUSH_BYTES && plan_len != reduce_ops::REQUEST_BYTES {
        set_error(format!(
            "reduce: the plan block is {plan_len} bytes, not {} or {}",
            reduce_ops::PUSH_BYTES,
            reduce_ops::REQUEST_BYTES
        ));
        return ACK_ERR_SIZE;
    }
    let bytes = unsafe { std::slice::from_raw_parts(plan, plan_len) };
    let mut block = [0u8; reduce_ops::PUSH_BYTES];
    block.copy_from_slice(&bytes[..reduce_ops::PUSH_BYTES]);
    // the 80-byte block is the call it always was and means f32 on both
    // sides; the trailing word, when present, is the storage request
    let storage = if plan_len == reduce_ops::REQUEST_BYTES {
        u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]])
    } else {
        0
    };
    if storage & !reduce_ops::STORAGE_FLAGS != 0 {
        set_error("reduce: plan refused: reduce-storage-flags-out-of-range".to_string());
        return ACK_ERR_SHAPE;
    }
    // the request is decoded before any device lookup, as the pointwise plan's operation is
    let request = match ReduceRequest::decode(&block) {
        Ok(r) => r,
        Err(name) => {
            set_error(format!("reduce: plan refused: {name}"));
            return ACK_ERR_SHAPE;
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    // mutable because the bf16 storage modules are built on the first call
    // that asks for one, and the guard cannot be borrowed mutably while the
    // buffers resolved below are alive
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // ALL SIX AT ONCE, deliberately. Which module each PASS needs depends
    // on the split count and on whether the plan picks v1 or v2 for its
    // first pass, and the plan cannot be built until the buffers are
    // resolved -- which is exactly when the guard can no longer be borrowed
    // mutably. Building the set costs at most six pipeline creations once
    // per device, and a device that never reduces a bf16 operand builds
    // none of them.
    if storage != 0 {
        let state: &mut Inner = &mut guard;
        for index in 0..3usize {
            for (slot, spv, push) in [
                (
                    &mut state.reduce_storage[index],
                    SPV_REDUCE_STORAGE[index],
                    reduce_ops::PUSH_BYTES as u32,
                ),
                (
                    &mut state.reduce_v2_storage[index],
                    SPV_REDUCE_V2_STORAGE[index],
                    reduce_ops::PUSH_BYTES_V2 as u32,
                ),
            ] {
                if slot.is_some() {
                    continue;
                }
                match state.ctx.kernel(spv, 2, push) {
                    Ok(k) => *slot = Some(k),
                    Err(e) => {
                        set_error(format!("reduce bf16 pipeline: {e}"));
                        return code_for(&e);
                    }
                }
            }
        }
    }
    let guard = guard;
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle; a null partials handle is a one-pass
    // caller's, decided once the plan says how many passes it needs
    let x = match buffer_of(&device, &guard, x) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let partials = if partials.is_null() {
        None
    } else {
        match buffer_of(&device, &guard, partials) {
            Ok(b) => Some(b),
            Err(code) => return code,
        }
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let view = match ReduceView::new(
        &request.shape[..request.ndim],
        &request.stride[..request.ndim],
        request.offset,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_error(format!("reduce: plan refused: {e}"));
            return reduce_plan_code(e);
        }
    };
    // the capacity is the real byte length in the INPUT's own elements
    let in_bytes = reduce_ops::element_bytes(storage & reduce_ops::FLAG_IN_BF16 != 0);
    let out_bytes = reduce_ops::element_bytes(storage & reduce_ops::FLAG_OUT_BF16 != 0);
    let plan = match ReducePlan::with_storage(
        view,
        request.reduced_mask,
        request.op,
        request.finish,
        request.scale,
        x.bytes / in_bytes,
        !guard.reduce_v1_only,
        storage,
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("reduce: plan refused: {e}"));
            return reduce_plan_code(e);
        }
    };
    // the runtime's buffer id is the allocation's identity (one id per live
    // buffer, so two handles naming one buffer agree)
    if out.id() == x.id() {
        set_error("reduce: plan refused: reduce-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    if out.bytes / out_bytes < plan.output_len() as u64 {
        // the detail precedes the name so a consumer reading the last
        // colon-separated word (the torch adapter's refusal histogram) gets the name
        set_error(format!(
            "reduce: an output of {} elements for {}: plan refused: reduce-output-too-small",
            out.bytes / out_bytes,
            plan.output_len()
        ));
        return ACK_ERR_SIZE;
    }
    let scratch = match (plan.second(), partials) {
        (None, _) => None,
        (Some(_), None) => {
            set_error(format!(
                "reduce: a two-pass plan ({} splits) needs a partials buffer",
                plan.splits()
            ));
            return ACK_ERR_NULL;
        }
        (Some(_), Some(p)) => {
            if p.id() == x.id() || p.id() == out.id() {
                set_error("reduce: plan refused: reduce-partials-alias");
                return ACK_ERR_SHAPE;
            }
            if p.bytes / 4 < plan.first().output_len() as u64 {
                set_error(format!(
                    "reduce: partials of {} elements for {}: plan refused: reduce-partials-too-small",
                    p.bytes / 4,
                    plan.first().output_len()
                ));
                return ACK_ERR_SIZE;
            }
            Some(p)
        }
    };
    // Defensive: ack_open fails when the pipeline cannot be built, so an
    // open device always carries the kernel; the branch names the state
    // rather than trusting it.
    let kernel_v1 = match guard.reduce.as_ref() {
        Some(k) => k,
        None => {
            set_error("reduce: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    let kernel_v2 = match guard.reduce_v2.as_ref() {
        Some(k) => k,
        None => {
            set_error("reduce: v2 kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    // Each pass binds the kernel its plan named -- v2 lays the kept span
    // across the lanes when the plan selected it, and the second pass is
    // always v1 -- and then the STORAGE MODULE that pass's own two bits name.
    // Module 0 is the f32 pair built at open; 1..3 were built above.
    let module_of = |pass: &reduce_ops::ReducePass| match (pass.kernel(), pass.storage_module()) {
        (reduce_ops::ReduceKernel::V1, 0) => Some(kernel_v1),
        (reduce_ops::ReduceKernel::V2, 0) => Some(kernel_v2),
        (reduce_ops::ReduceKernel::V1, i) => guard.reduce_storage[i - 1].as_ref(),
        (reduce_ops::ReduceKernel::V2, i) => guard.reduce_v2_storage[i - 1].as_ref(),
    };
    let kernel = match module_of(plan.first()) {
        Some(k) => k,
        None => {
            set_error("reduce: storage module absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    let first_target = scratch.unwrap_or(out);
    if let Err(e) = guard.ctx.bind(kernel, &[x, first_target]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let first_access = [Access::read(x.id()), Access::write(first_target.id())];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        plan.first().push_constants(),
        plan.first().dispatch_groups(),
        1,
        &first_access,
    ));
    if rc != ACK_OK {
        return rc;
    }
    if let (Some(second), Some(p)) = (plan.second(), scratch) {
        // THE TWO PASSES ARE RECORDED INTO ONE BATCH. Rebinding between them
        // is safe because each recorded dispatch takes its OWN descriptor set
        // from the kernel's arena and `bind` writes no set at all; the read of
        // `partials` the second pass makes is ordered after the first pass's
        // write by R1's barrier. Before deferred submission this was safe for
        // a different reason -- the first pass had already been submitted and
        // waited for -- and that reason no longer holds.
        // the second pass reads f32 partials and writes the caller's output,
        // so its module is the output bit alone -- a different module from the
        // first pass's whenever the caller asked for bf16 on both sides
        let second_kernel = match module_of(second) {
            Some(k) => k,
            None => {
                set_error("reduce: storage module absent on this device");
                return ACK_ERR_UNSUPPORTED_OP;
            }
        };
        if let Err(e) = guard.ctx.bind(second_kernel, &[p, out]) {
            set_error(format!("{e}"));
            return code_for(&e);
        }
        let second_access = [Access::read(p.id()), Access::write(out.id())];
        let rc = recorded(guard.ctx.dispatch(
            second_kernel,
            second.push_constants(),
            second.dispatch_groups(),
            1,
            &second_access,
        ));
        if rc != ACK_OK {
            return rc;
        }
    }
    // recorded, not run: no duration exists for this call to report
    unsafe { no_duration(ms_out) };
    ACK_OK
}

/// Which of the row kernel's input slots an operation reads (the kernel's own
/// law, rowwise_f32.comp): every operation reads `primary`; the backwards
/// read `upstream`; RMSNorm reads `gamma`. A slot an operation does not read
/// may be bound to any live buffer of the device.
fn rowwise_reads(op: RowOp) -> (bool, bool) {
    let upstream = matches!(
        op,
        RowOp::SoftmaxBackward | RowOp::RmsNormBackward | RowOp::LogSoftmaxBackward
    );
    let gamma = matches!(op, RowOp::RmsNormForward | RowOp::RmsNormBackward);
    (upstream, gamma)
}

fn rowwise_plan_code(err: RowPlanError) -> i32 {
    match err {
        RowPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// One row operation of `row_ops::RowOp` over `rows` contiguous rows of `cols`
/// FP32 elements on the device (rowwise-f32/v2 at the C ABI, schema 1):
/// softmax forward (0) and backward (1, `primary` = the saved probabilities),
/// RMSNorm forward (2), backward (3, writing `2 * rows * cols`: dx then the
/// per-element gamma contributions) and its gamma reduction (4, reading the
/// backward's `2 * rows * cols` and writing `cols`), log-softmax forward (5)
/// and backward (6, `primary` = the saved log-probabilities); masked logits
/// (-inf entries) are admitted on the forwards. `plan` is the kernel's own
/// 16-byte push block: rows, cols, the operation word, the RMSNorm epsilon
/// bits (ignored by the other operations, which pass any finite value).
/// `row_ops::RowPlan::new` re-derives every length and grid here: an
/// operation word outside the family, a zero or over-limit dimension
/// (`row_ops::MAX_ROWS`, `MAX_COLS`), more than `MAX_ELEMENTS` elements and
/// an RMSNorm epsilon outside its range are refused by name before any
/// device lookup. The buffers are then checked against the plan: `primary`
/// must hold `primary_len()` elements, `upstream` `upstream_len()` when the
/// operation reads it, `gamma` `gamma_len()` when the operation reads it,
/// `out` `output_len()`, and `out` must be a buffer other than the three
/// inputs' (the kernel writes rows while other workgroups read them). The
/// plan module's `validate_inputs` conditions on the DATA (finite upstream
/// and gamma, no NaN or +inf logit, no fully masked row, probabilities in
/// [0, 1] summing to one, log-probabilities at most 0) are a host-side check
/// and are NOT re-validated here: the data lives on the device, and a
/// violation produces the kernel's own result (NaN for a fully masked row,
/// as torch's CPU kernel produces), never a refusal. Refusals:
/// `ACK_ERR_NULL` for a null block, `ACK_ERR_SIZE` for a block of another
/// length, an element count above the family's limit
/// (`rowwise-element-limit-exceeded`) or a buffer with too few elements
/// (`rowwise-primary-too-small`, `rowwise-upstream-too-small`,
/// `rowwise-gamma-too-small`, `rowwise-output-too-small`), `ACK_ERR_SHAPE`
/// for the plan module's other names and for an output that is an input's
/// buffer (`rowwise-output-aliases-input`). The dispatch is RECORDED and runs
/// at the next flush, so `ms_out` is NaN: no duration exists for this call to
/// report (increment 5, deferred submission).
/// # Safety
/// `plan` must point to `plan_len` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_rowwise(
    dev: *const AckDevice,
    primary: *const Buffer,
    upstream: *const Buffer,
    gamma: *const Buffer,
    out: *const Buffer,
    plan: *const u8,
    plan_len: usize,
    ms_out: *mut f64,
) -> i32 {
    if plan.is_null() {
        return ACK_ERR_NULL;
    }
    if plan_len != ROWWISE_PUSH_BYTES {
        set_error(format!(
            "rowwise: the plan block is {plan_len} bytes, not {ROWWISE_PUSH_BYTES}"
        ));
        return ACK_ERR_SIZE;
    }
    let mut block = [0u8; ROWWISE_PUSH_BYTES];
    block.copy_from_slice(unsafe { std::slice::from_raw_parts(plan, plan_len) });
    let word = |i: usize| {
        u32::from_le_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ])
    };
    // the operation word and the plan are decoded before any device lookup
    let op = match RowOp::try_from(word(2)) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("rowwise: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match RowPlan::new(word(0), word(1), op, f32::from_bits(word(3))) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("rowwise: plan refused: {e}"));
            return rowwise_plan_code(e);
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let primary = match buffer_of(&device, &guard, primary) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let upstream = match buffer_of(&device, &guard, upstream) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let gamma = match buffer_of(&device, &guard, gamma) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let (reads_upstream, reads_gamma) = rowwise_reads(op);
    // the detail precedes the name so a consumer reading the last
    // colon-separated word (the torch adapter's refusal histogram) gets the name
    let capacity = |buffer: &Buffer, needed: usize, name: &str| -> Result<(), i32> {
        if buffer.bytes / 4 < needed as u64 {
            set_error(format!(
                "rowwise: {name} of {} elements for {needed}: plan refused: rowwise-{name}-too-small",
                buffer.bytes / 4
            ));
            return Err(ACK_ERR_SIZE);
        }
        Ok(())
    };
    if let Err(code) = capacity(primary, plan.primary_len(), "primary") {
        return code;
    }
    if reads_upstream {
        if let Err(code) = capacity(upstream, plan.upstream_len(), "upstream") {
            return code;
        }
    }
    if reads_gamma {
        if let Err(code) = capacity(gamma, plan.gamma_len(), "gamma") {
            return code;
        }
    }
    if let Err(code) = capacity(out, plan.output_len(), "output") {
        return code;
    }
    // the runtime's buffer id is the allocation's identity: an output that is
    // an input's buffer would be read by other workgroups while written
    if out.id() == primary.id()
        || (reads_upstream && out.id() == upstream.id())
        || (reads_gamma && out.id() == gamma.id())
    {
        set_error("rowwise: plan refused: rowwise-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    // Defensive: ack_open fails when the pipeline cannot be built, so an
    // open device always carries the kernel; the branch names the state
    // rather than trusting it.
    let kernel = match guard.rowwise.as_ref() {
        Some(k) => k,
        None => {
            set_error("rowwise: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[primary, upstream, gamma, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // `upstream` and `gamma` are bound to a live buffer even for an operation
    // that reads neither (`rowwise_reads` is the kernel's own law); an unused
    // slot is declared Read
    let access = [
        Access::read(primary.id()),
        Access::read(upstream.id()),
        Access::read(gamma.id()),
        Access::write(out.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// Optional vq4 extension version, separate from the unchanged ABI10 struct.
/// A binding must resolve this symbol and require exactly 1 before ack_vq.
#[no_mangle]
pub extern "C" fn ack_vq_schema() -> u32 {
    crate::vq_ops::SCHEMA
}

fn vq_plan_code(error: VqPlanError) -> i32 {
    match error {
        VqPlanError::Capacity(_) | VqPlanError::AddressLimit | VqPlanError::WorkLimit => {
            ACK_ERR_SIZE
        }
        _ => ACK_ERR_SHAPE,
    }
}

/// vq4/v1 encode(0), decode(1), packed-pair matmul(2), or decoded-state
/// AdamW(3). The 32-byte plan is [op,m,n,k,group,16,flags,0] as little-endian
/// u32 words. See vq_ops and vq4.comp for the closed per-operation layout.
/// All six handles must be live on this device, including unread inputs.
/// Output and status must be distinct from every input and each other.
/// Status is a caller-cleared u32: bits 1/2/4 mean nonfinite arithmetic,
/// defensive layout failure, and invalid optimizer coefficients/moments.
/// ACK_OK means recorded; download and check status before accepting output.
/// No persistent FP32 optimizer master is created by this entry.
///
/// Admission derives sizes from the plan and the registry's allocation byte
/// capacities under one device lock, before even creating the lazy kernel.
/// Plan/size/alias/grid refusals cannot enqueue work or mutate buffers.
/// # Safety
/// `plan` must point to `plan_len` readable bytes when non-null. Handles are
/// registry ids; this entry never dereferences a caller-supplied handle.
#[no_mangle]
pub unsafe extern "C" fn ack_vq(
    dev: *const AckDevice,
    b0: *const Buffer,
    b1: *const Buffer,
    b2: *const Buffer,
    b3: *const Buffer,
    out: *const Buffer,
    status: *const Buffer,
    plan: *const u8,
    plan_len: usize,
) -> i32 {
    if plan.is_null() {
        set_error("vq: null plan");
        return ACK_ERR_NULL;
    }
    if plan_len != VQ_PUSH_BYTES {
        set_error(format!(
            "vq: plan has {plan_len} bytes, expected {VQ_PUSH_BYTES}"
        ));
        return ACK_ERR_SIZE;
    }
    let bytes = unsafe { std::slice::from_raw_parts(plan, plan_len) };
    let mut words = [0u32; 8];
    for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
        *word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let plan = match VqPlan::from_words(words) {
        Ok(plan) => plan,
        Err(error) => {
            set_error(format!("vq: plan refused: {error}"));
            return vq_plan_code(error);
        }
    };
    let device = match device(dev) {
        Ok(device) => device,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(guard) => guard,
        Err(code) => return code,
    };
    if guard.ctx.is_poisoned() {
        set_error("vq: device poisoned by an earlier ambiguous operation");
        return ACK_ERR_POISONED;
    }
    let handles = [b0, b1, b2, b3, out, status];
    let mut capacities = [0u64; crate::vq_ops::OPERANDS];
    let mut ids = [0u64; crate::vq_ops::OPERANDS];
    for (i, &handle) in handles.iter().enumerate() {
        let buffer = match buffer_of(&device, &guard, handle) {
            Ok(buffer) => buffer,
            Err(code) => return code,
        };
        capacities[i] = buffer.bytes;
        ids[i] = buffer.id();
    }
    if let Err(error) = plan
        .validate_buffers(capacities, ids)
        .and_then(|()| plan.validate_dispatch(guard.ctx.max_workgroup_count))
    {
        set_error(format!("vq: plan refused: {error}"));
        return vq_plan_code(error);
    }
    if guard.vq.is_none() {
        match guard
            .ctx
            .kernel(SPV_VQ, crate::vq_ops::OPERANDS as u32, VQ_PUSH_BYTES as u32)
        {
            Ok(kernel) => guard.vq = Some(kernel),
            Err(error) => {
                set_error(format!("vq: pipeline creation refused: {error}"));
                return code_for(&error);
            }
        }
    }
    // Registry membership cannot change while this same device lock is held.
    // Resolve again to keep kernel initialization outside the borrow lifetime.
    let buffers: Result<Vec<_>, _> = handles
        .iter()
        .map(|&handle| buffer_of(&device, &guard, handle))
        .collect();
    let buffers = match buffers {
        Ok(buffers) => buffers,
        Err(code) => return code,
    };
    let kernel = match guard.vq.as_ref() {
        Some(kernel) => kernel,
        None => {
            set_error("vq: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(error) = guard.ctx.bind(kernel, &buffers) {
        set_error(format!("vq: binding refused: {error}"));
        return code_for(&error);
    }
    // Write includes the status word's atomic read/modify/write. The canonical
    // recorder supplies hazards/barriers against earlier reads and writes.
    let access = [
        Access::read(ids[0]),
        Access::read(ids[1]),
        Access::read(ids[2]),
        Access::read(ids[3]),
        Access::write(ids[4]),
        Access::write(ids[5]),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// Which of the loss kernel's buffers an operation READS.
///
/// THE KERNEL IS AUTHORITATIVE for this at the C ABI, and that is a law rather
/// than a workaround. `LossPlan::primary_len` answers a different question: how
/// long the primary array must be for `validate_inputs`, the plan's HOST-side
/// data check, which conservatively inspects the log-probabilities for
/// `NllBackward` even though `kernels/loss_f32.comp` never reads them on that
/// branch. A C caller has no host array to validate -- its data is already on
/// the device -- so the only question left is which buffers the dispatch
/// dereferences, and that is the kernel's answer, not the plan's.
///
/// The practical consequence is why it matters: the NLL backward has no logits
/// to bind, so requiring a buffer for it would be a refusal with nothing behind
/// it. A slot an operation does not read may be any live buffer, as in the
/// pointwise and row families. The test named for this law is what defends it
/// if a future kernel change starts reading `primary` on that branch.
fn loss_reads(op: LossOp) -> (bool, bool) {
    let primary = op != LossOp::NllBackward;
    let upstream = matches!(op, LossOp::Backward | LossOp::NllBackward);
    (primary, upstream)
}

fn loss_plan_code(err: LossPlanError) -> i32 {
    match err {
        LossPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// One operation of `loss_ops::LossOp` over `rows` rows of `cols` FP32 logits
/// (indexed-cross-entropy-f32/v2 at the C ABI, schema 1): cross-entropy
/// forward (0) and backward (1), the reduction of per-row losses (2), and the
/// negative log-likelihood of log-probabilities forward (3) and backward (4).
///
/// This entry does NOT take an opaque push block, and that is deliberate.
/// `LossPlan` derives host data: it canonicalises each target from the
/// caller's i64 domain to the kernel's i32, refuses one outside `[-1, cols)`
/// BY NAME, maps the caller's `ignore_index` to -1, and counts the non-ignored
/// targets to build the mean normalizer. Handing that work to the caller and
/// trusting the result would turn a named refusal into an undetected
/// corruption, so the caller passes the parameters and the host targets, and
/// the entry validates them, UPLOADS the plan's own canonical image into
/// `targets`, and dispatches. The upload is the kit's, so a caller that counts
/// transfers must count `rows * 4` bytes for it; the torch adapter does.
///
/// `targets` must therefore be a device buffer of at least `rows` i32
/// elements, and its previous contents are overwritten. `uploaded_bytes`, when
/// not null, receives the number of bytes this entry uploaded, so a caller
/// that accounts for transfers reports a MEASURED figure rather than its own
/// arithmetic about one; it is written on every path that uploads, and left
/// untouched on every path that refuses before uploading. `primary` holds the
/// logits (or the per-row losses for operation 2, or the log-probabilities for
/// 3 and 4); `upstream` the incoming gradient, one value per row under
/// reduction `None` and a single value otherwise. Buffers an operation does
/// not read are unconstrained (see `loss_reads`), and `out` may not be any
/// buffer the operation reads.
///
/// Refusals are by name before any dispatch: `ACK_ERR_NULL` for a null target
/// pointer with a nonzero length, `ACK_ERR_SHAPE` for every name
/// `loss_ops::LossPlanError` raises (an unknown operation or reduction, a zero
/// or over-limit dimension, a target count that is not `rows`, a target out of
/// range, a mean with no valid targets) and for `loss-output-aliases-input`,
/// and `ACK_ERR_SIZE` for the element limit or a buffer too small
/// (`loss-primary-too-small`, `loss-upstream-too-small`,
/// `loss-output-too-small`), and `loss-targets-size-mismatch` for a targets
/// buffer that is not EXACTLY the plan's image, since this entry writes that
/// one and the kit uploads a whole buffer. The plan module's
/// FINITENESS conditions are not re-validated here: that data is on the
/// device, so a nonfinite logit produces the kernel's own result, as it does
/// for the row family. The dispatch is RECORDED and runs
/// at the next flush, so `ms_out` is NaN: no duration exists for this call to
/// report (increment 5, deferred submission).
/// # Safety
/// `targets` must point to `targets_len` readable `i64` values; `ms_out` and
/// `uploaded_bytes` may be null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_loss(
    dev: *const AckDevice,
    primary: *const Buffer,
    targets: *const Buffer,
    upstream: *const Buffer,
    out: *const Buffer,
    rows: u32,
    cols: u32,
    op: u32,
    reduction: u32,
    ignore_index: i64,
    target_values: *const i64,
    targets_len: usize,
    uploaded_bytes: *mut u64,
    ms_out: *mut f64,
) -> i32 {
    if target_values.is_null() && targets_len != 0 {
        return ACK_ERR_NULL;
    }
    // the operation, the reduction and the whole plan are decoded and validated
    // before any device lookup, as every other family in this file does
    let op = match LossOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("loss: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let reduction = match LossReduction::try_from(reduction) {
        Ok(r) => r,
        Err(e) => {
            set_error(format!("loss: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let values: &[i64] = if targets_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(target_values, targets_len) }
    };
    let plan = match LossPlan::new(rows, cols, op, reduction, ignore_index, values) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("loss: plan refused: {e}"));
            return loss_plan_code(e);
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let primary = match buffer_of(&device, &guard, primary) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let targets = match buffer_of(&device, &guard, targets) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let upstream = match buffer_of(&device, &guard, upstream) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let (reads_primary, reads_upstream) = loss_reads(op);
    // the detail precedes the name so a consumer reading the last
    // colon-separated word (the torch adapter's refusal histogram) gets the name
    let capacity = |buffer: &Buffer, needed: usize, name: &str| -> Result<(), i32> {
        if buffer.bytes / 4 < needed as u64 {
            set_error(format!(
                "loss: {name} of {} elements for {needed}: plan refused: loss-{name}-too-small",
                buffer.bytes / 4
            ));
            return Err(ACK_ERR_SIZE);
        }
        Ok(())
    };
    if reads_primary {
        if let Err(code) = capacity(primary, plan.primary_len(), "primary") {
            return code;
        }
    }
    // The targets buffer is the one this entry WRITES, and `Context::upload`
    // moves a whole buffer -- it refuses `data.len() != dst.bytes`. So this slot
    // is not "at least" like the others: it must be exactly the plan's image.
    // Checking it here turns what would otherwise surface as the kit's opaque
    // `unsupported: upload of N bytes into an M-byte buffer` into a name, which
    // is what a refusal histogram can carry. Found by the ABI test, not by
    // reading the upload path.
    if targets.bytes != (plan.targets_len() * 4) as u64 {
        set_error(format!(
            "loss: targets buffer of {} bytes for an image of {}: plan refused: loss-targets-size-mismatch",
            targets.bytes,
            plan.targets_len() * 4
        ));
        return ACK_ERR_SIZE;
    }
    if reads_upstream {
        if let Err(code) = capacity(upstream, plan.upstream_len(), "upstream") {
            return code;
        }
    }
    if let Err(code) = capacity(out, plan.output_len(), "output") {
        return code;
    }
    // the runtime's buffer id is the allocation's identity: the kernel writes
    // the output while other workgroups still read the inputs, and it also
    // writes the targets image below, so the output may be neither
    if out.id() == targets.id()
        || (reads_primary && out.id() == primary.id())
        || (reads_upstream && out.id() == upstream.id())
    {
        set_error("loss: plan refused: loss-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    // The plan's OWN canonical image is what the kernel reads: uploading it
    // here is what makes the validation above binding on the dispatch, rather
    // than a statement about bytes the caller promised to have written.
    let image = plan.target_bytes();
    if let Err(e) = guard.ctx.upload(targets, &image) {
        set_error(format!("loss: uploading the validated targets: {e}"));
        return code_for(&e);
    }
    // Reported, not assumed. A caller that counts transfers measures this
    // instead of restating rows * 4 in its own arithmetic, which would drift
    // the first time the image changes shape (padding, an alignment round-up,
    // a second buffer) with nothing to catch it.
    if !uploaded_bytes.is_null() {
        unsafe { *uploaded_bytes = image.len() as u64 };
    }
    // Defensive: ack_open fails when the pipeline cannot be built, so an
    // open device always carries the kernel; the branch names the state
    // rather than trusting it.
    let kernel = match guard.loss.as_ref() {
        Some(k) => k,
        None => {
            set_error("loss: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[primary, targets, upstream, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // `upstream` is bound to a live buffer even for an operation that does not
    // read it (`loss_reads` is the kernel's own law); an unused slot is
    // declared Read. The targets image was uploaded above, and that upload
    // flushed everything recorded before it, so the read of `targets` this
    // dispatch makes is ordered after the copy that wrote it.
    let access = [
        Access::read(primary.id()),
        Access::read(targets.id()),
        Access::read(upstream.id()),
        Access::write(out.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

fn embedding_plan_code(err: EmbeddingPlanError) -> i32 {
    match err {
        EmbeddingPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// The embedding gather (0) and its dense backward (1) over FP32 rows of `dim`
/// (embedding-f32/v2 at the C ABI, schema 1).
///
/// Like `ack_loss` and unlike the earlier families, this entry takes the
/// caller's HOST ids rather than an opaque plan block, because
/// `embedding_ops::EmbeddingPlan` derives host data from them: it validates
/// every id in the caller's i64 domain before any narrowing, and for the
/// backward it builds the CSR — bucket offsets by vocabulary id and token
/// positions grouped within each bucket in original order — which is what lets
/// the backward accumulate without atomics. The entry validates, uploads BOTH
/// derived images (`indices` and `offsets`), and dispatches, so the validation
/// is binding on the dispatch rather than a statement about bytes the caller
/// promised to have written. `uploaded_bytes`, when not null, receives the
/// total it wrote, so a caller accounting for transfers measures rather than
/// restating arithmetic.
///
/// The two images differ by operation, which the plan decides and this entry
/// does not second-guess: the gather's index image is the ids and its offset
/// image is a single unused word, while the backward's index image is the
/// grouped POSITIONS and its offset image is the `vocab + 1` bucket offsets.
/// Both buffers must therefore be EXACTLY their image, since the kit uploads a
/// whole buffer; `primary` and `out` are "at least", as in the other families.
///
/// `primary` is the weight table for the gather and the upstream token
/// gradients for the backward; `out` is the token vectors or the dense
/// vocabulary gradient. Refusals are by name before any dispatch:
/// `ACK_ERR_NULL` for a null id pointer with a nonzero length, `ACK_ERR_SHAPE`
/// for every name `EmbeddingPlanError` raises and for
/// `embedding-output-aliases-input`, `ACK_ERR_SIZE` for the element limit or a
/// buffer that is not its image (`embedding-indices-size-mismatch`,
/// `embedding-offsets-size-mismatch`) or too small
/// (`embedding-primary-too-small`, `embedding-output-too-small`). The plan's
/// finiteness check over `primary` is not re-validated here: that data is on
/// the device. The dispatch is RECORDED and runs
/// at the next flush, so `ms_out` is NaN: no duration exists for this call to
/// report (increment 5, deferred submission).
/// # Safety
/// `ids` must point to `ids_len` readable `i64` values; `ms_out` and
/// `uploaded_bytes` may be null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_embedding(
    dev: *const AckDevice,
    primary: *const Buffer,
    indices: *const Buffer,
    offsets: *const Buffer,
    out: *const Buffer,
    n: u32,
    vocab: u32,
    dim: u32,
    op: u32,
    ids: *const i64,
    ids_len: usize,
    uploaded_bytes: *mut u64,
    ms_out: *mut f64,
) -> i32 {
    if ids.is_null() && ids_len != 0 {
        return ACK_ERR_NULL;
    }
    let op = match EmbeddingOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("embedding: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let values: &[i64] = if ids_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ids, ids_len) }
    };
    let plan = match EmbeddingPlan::new(n, vocab, dim, op, values) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("embedding: plan refused: {e}"));
            return embedding_plan_code(e);
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let primary = match buffer_of(&device, &guard, primary) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let indices = match buffer_of(&device, &guard, indices) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let offsets = match buffer_of(&device, &guard, offsets) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let index_image = plan.index_bytes();
    let offset_image = plan.offset_bytes();
    // the detail precedes the name so a consumer reading the last
    // colon-separated word gets the name
    let exact = |buffer: &Buffer, image: usize, name: &str| -> Result<(), i32> {
        if buffer.bytes != image as u64 {
            set_error(format!(
                "embedding: {name} buffer of {} bytes for an image of {image}: plan refused: embedding-{name}-size-mismatch",
                buffer.bytes
            ));
            return Err(ACK_ERR_SIZE);
        }
        Ok(())
    };
    let capacity = |buffer: &Buffer, needed: usize, name: &str| -> Result<(), i32> {
        if buffer.bytes / 4 < needed as u64 {
            set_error(format!(
                "embedding: {name} of {} elements for {needed}: plan refused: embedding-{name}-too-small",
                buffer.bytes / 4
            ));
            return Err(ACK_ERR_SIZE);
        }
        Ok(())
    };
    if let Err(code) = exact(indices, index_image.len(), "indices") {
        return code;
    }
    if let Err(code) = exact(offsets, offset_image.len(), "offsets") {
        return code;
    }
    if let Err(code) = capacity(primary, plan.primary_len(), "primary") {
        return code;
    }
    if let Err(code) = capacity(out, plan.output_len(), "output") {
        return code;
    }
    // the kernel writes the output while other workgroups read the inputs, and
    // this entry writes both index images, so the output may be none of them
    if out.id() == primary.id() || out.id() == indices.id() || out.id() == offsets.id() {
        set_error("embedding: plan refused: embedding-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    // the plan's OWN images are what the kernel reads
    if let Err(e) = guard.ctx.upload(indices, &index_image) {
        set_error(format!("embedding: uploading the validated indices: {e}"));
        return code_for(&e);
    }
    if let Err(e) = guard.ctx.upload(offsets, &offset_image) {
        set_error(format!("embedding: uploading the validated offsets: {e}"));
        return code_for(&e);
    }
    if !uploaded_bytes.is_null() {
        unsafe { *uploaded_bytes = (index_image.len() + offset_image.len()) as u64 };
    }
    let kernel = match guard.embedding.as_ref() {
        Some(k) => k,
        None => {
            set_error("embedding: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[primary, indices, offsets, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // both index images were uploaded above, and each upload flushed
    // everything recorded before it, so the reads this dispatch makes are
    // ordered after the copies that wrote them
    let access = [
        Access::read(primary.id()),
        Access::read(indices.id()),
        Access::read(offsets.id()),
        Access::write(out.id()),
    ];
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ));
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ambiguous_timing_has_its_own_code_and_a_poison_keeps_its_own() {
        assert_eq!(
            code_for(&AckError::AmbiguousTiming("x".into())),
            ACK_ERR_TIMING
        );
        assert_eq!(code_for(&AckError::Poisoned("x".into())), ACK_ERR_POISONED);
        assert_ne!(ACK_ERR_TIMING, ACK_ERR_SHAPE);
    }

    /// The `ms_out` contract of a RECORDED dispatch, on the host.
    ///
    /// Every operator entry in this file calls `no_duration` on its `ACK_OK`
    /// path, so this is the one place the value is produced. NaN is asserted
    /// exactly, not `is_nan() || >= 0.0`: a regression to 0.0 would read to a
    /// caller as an instantaneous dispatch, and a regression to a stale value
    /// would read as a duration of work that has not run. The Python surface
    /// pins the other end of the same contract
    /// (`tests/languages/test_compute_kit_ffi.py`), on a device.
    #[test]
    fn a_recorded_dispatch_reports_nan_not_zero_and_tolerates_a_null_out() {
        let mut ms = 12.5f64;
        unsafe { no_duration(&mut ms) };
        assert!(
            ms.is_nan(),
            "a recorded dispatch has no duration; ms_out must be NaN, got {ms}"
        );
        assert_ne!(
            ms.to_bits(),
            0.0f64.to_bits(),
            "0.0 is a duration, NaN is not"
        );
        // `ms_out` is documented as nullable at every entry
        unsafe { no_duration(std::ptr::null_mut()) };
    }

    #[test]
    fn buffer_handles_name_their_owner_and_never_collide_across_devices() {
        let h = buffer_handle(7, 3);
        assert_eq!(split_buffer_handle(h), (7, 3));
        assert_ne!(buffer_handle(7, 3), buffer_handle(8, 3));
        assert_ne!(buffer_handle(7, 3), buffer_handle(7, 4));
        assert_ne!(buffer_handle(1, 1), 0, "a live handle is never null");
        assert_eq!(
            split_buffer_handle(buffer_handle(u32::MAX as u64, u32::MAX)),
            (u32::MAX as u64, u32::MAX)
        );
    }

    #[test]
    fn unknown_handles_are_refused_by_name_without_a_device() {
        // no device is open in this test process; every code here comes from the
        // registry, never from Vulkan
        let bogus = 0x7ACC_0001usize as *mut AckDevice;
        assert_eq!(ack_close(bogus), ACK_ERR_CLOSED);
        assert_eq!(ack_close(std::ptr::null_mut()), ACK_ERR_NULL);
        assert!(ack_buffer_alloc(bogus, 64).is_null());
        assert_eq!(
            ack_buffer_free(bogus, 0x1_0000_0001usize as *mut Buffer),
            ACK_ERR_CLOSED
        );
        let mut out = [0u8; 8];
        assert_eq!(
            unsafe { ack_upload(bogus, 0x1_0000_0001usize as *const Buffer, out.as_ptr(), 8) },
            ACK_ERR_CLOSED
        );
        assert_eq!(
            unsafe {
                ack_download(
                    bogus,
                    0x1_0000_0001usize as *const Buffer,
                    out.as_mut_ptr(),
                    8,
                )
            },
            ACK_ERR_CLOSED
        );
        assert_eq!(
            ack_cast(
                bogus,
                0x1_0000_0001usize as *const Buffer,
                0x1_0000_0002usize as *const Buffer,
                4,
                0
            ),
            ACK_ERR_CLOSED
        );
        // a well-formed pointwise block reaches the registry, which refuses the handle
        let block = [0u8; pointwise_ops::PUSH_BYTES];
        assert_eq!(
            unsafe {
                ack_pointwise(
                    bogus,
                    0x1_0000_0001usize as *const Buffer,
                    0x1_0000_0001usize as *const Buffer,
                    0x1_0000_0001usize as *const Buffer,
                    0x1_0000_0002usize as *const Buffer,
                    block.as_ptr(),
                    block.len(),
                    std::ptr::null_mut(),
                )
            },
            ACK_ERR_CLOSED
        );
        let mut name = [0 as c_char; 16];
        assert_eq!(
            unsafe { ack_device_name(bogus, name.as_mut_ptr(), 16) },
            ACK_ERR_CLOSED
        );
        let mut msg = [0 as c_char; 128];
        assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
        let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
        assert!(text.contains("not open"), "{text}");
    }

    #[test]
    fn abi_info_reports_the_version_and_refuses_a_short_struct() {
        let mut info: AckAbiInfo = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { ack_abi_info(&mut info, std::mem::size_of::<AckAbiInfo>()) },
            ACK_OK
        );
        assert_eq!(info.version, ACK_ABI_VERSION);
        assert_eq!(info.size as usize, std::mem::size_of::<AckAbiInfo>());
        assert_eq!(
            info.operators,
            ACK_OP_MATMUL_BF16
                | ACK_OP_CAST
                | ACK_OP_MATMUL_F32
                | ACK_OP_POINTWISE
                | ACK_OP_REDUCE
                | ACK_OP_ROWWISE
                | ACK_OP_LOSS
                | ACK_OP_EMBEDDING
        );
        assert_eq!(info.op_schema_matmul_f32, ACK_OP_SCHEMA_MATMUL_F32);
        assert_eq!(info.op_schema_pointwise, ACK_OP_SCHEMA_POINTWISE);
        assert_eq!(info.op_schema_reduce, ACK_OP_SCHEMA_REDUCE);
        assert_eq!(info.op_schema_rowwise, ACK_OP_SCHEMA_ROWWISE);
        assert_eq!(info.op_schema_loss, ACK_OP_SCHEMA_LOSS);
        assert_eq!(info.op_schema_embedding, ACK_OP_SCHEMA_EMBEDDING);
        assert_eq!(info.shaders, shader_identity());
        assert_ne!(info.shaders, 0);
        assert_eq!(info.build, build_identity());
        assert_eq!((info.op_schema_matmul_bf16, info.op_schema_cast), (2, 1));
        assert_eq!(
            (
                info.capability_streams,
                info.capability_rng,
                info.capability_optimizer,
                info.capability_serialization
            ),
            (0, 0, 0, 0),
            "absent capabilities are 0, never a version"
        );
        assert_eq!(
            unsafe { ack_abi_info(&mut info, 24) },
            ACK_ERR_ABI,
            "a version-1 caller's struct is refused"
        );
        assert_eq!(
            unsafe { ack_abi_info(std::ptr::null_mut(), 64) },
            ACK_ERR_NULL
        );
    }

    #[test]
    fn the_structs_have_the_sizes_and_offsets_the_ctypes_mirrors_assume() {
        // ack.py and ack_torch.py declare these layouts field by field; a change
        // here must change them too, and the handshake requires exact sizes.
        // The offsets below are the same table tests/languages/test_compute_kit_handshake.py
        // pins against both ctypes mirrors, so a same-width reorder fails somewhere.
        assert_eq!(std::mem::size_of::<AckAbiInfo>(), 80);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_matmul_f32), 56);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_pointwise), 60);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_reduce), 64);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_embedding), 76);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, shaders), 16);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, build), 24);
        assert_eq!(
            std::mem::offset_of!(AckAbiInfo, capability_serialization),
            52
        );
        assert_eq!(std::mem::size_of::<AckDeviceInfo>(), 616);
        assert_eq!(std::mem::align_of::<AckDeviceInfo>(), 8);
        for (name, offset) in [
            (
                "device_uuid",
                std::mem::offset_of!(AckDeviceInfo, device_uuid),
            ),
            (
                "driver_uuid",
                std::mem::offset_of!(AckDeviceInfo, driver_uuid),
            ),
            ("features", std::mem::offset_of!(AckDeviceInfo, features)),
            (
                "timestamp_period_ns",
                std::mem::offset_of!(AckDeviceInfo, timestamp_period_ns),
            ),
            (
                "device_local_bytes",
                std::mem::offset_of!(AckDeviceInfo, device_local_bytes),
            ),
            (
                "max_workgroup_count",
                std::mem::offset_of!(AckDeviceInfo, max_workgroup_count),
            ),
            (
                "matmul_tile",
                std::mem::offset_of!(AckDeviceInfo, matmul_tile),
            ),
            ("identity", std::mem::offset_of!(AckDeviceInfo, identity)),
            (
                "device_name",
                std::mem::offset_of!(AckDeviceInfo, device_name),
            ),
            (
                "driver_name",
                std::mem::offset_of!(AckDeviceInfo, driver_name),
            ),
            (
                "driver_info",
                std::mem::offset_of!(AckDeviceInfo, driver_info),
            ),
        ] {
            let expected = match name {
                "device_uuid" => 32,
                "driver_uuid" => 48,
                "features" => 64,
                "timestamp_period_ns" => 80,
                "device_local_bytes" => 88,
                "max_workgroup_count" => 108,
                "matmul_tile" => 132,
                "identity" => 160,
                "device_name" => 168,
                "driver_name" => 424,
                "driver_info" => 488,
                _ => unreachable!(),
            };
            assert_eq!(offset, expected, "{name}");
        }
        assert_eq!(
            DEVICE_INFO_HASHED,
            8..160,
            "the hashed range ends where identity begins"
        );
        // The dispatch instrument's struct. It is NOT in the extension's
        // pointer table and the ABI version did not move for it, so its own
        // size and version ARE its handshake: this pin is what a ctypes mirror
        // written for version 1 is entitled to rely on.
        assert_eq!(std::mem::size_of::<AckDispatchStats>(), 96);
        assert_eq!(std::mem::align_of::<AckDispatchStats>(), 8);
        for (name, offset) in [
            ("size", std::mem::offset_of!(AckDispatchStats, size)),
            ("version", std::mem::offset_of!(AckDispatchStats, version)),
            ("items", std::mem::offset_of!(AckDispatchStats, items)),
            ("commands", std::mem::offset_of!(AckDispatchStats, commands)),
            ("barriers", std::mem::offset_of!(AckDispatchStats, barriers)),
            ("flushes", std::mem::offset_of!(AckDispatchStats, flushes)),
            (
                "timed_flushes",
                std::mem::offset_of!(AckDispatchStats, timed_flushes),
            ),
            (
                "timed_items",
                std::mem::offset_of!(AckDispatchStats, timed_items),
            ),
            (
                "device_ns",
                std::mem::offset_of!(AckDispatchStats, device_ns),
            ),
            (
                "min_item_ns",
                std::mem::offset_of!(AckDispatchStats, min_item_ns),
            ),
            (
                "max_item_ns",
                std::mem::offset_of!(AckDispatchStats, max_item_ns),
            ),
            (
                "unresolved_flushes",
                std::mem::offset_of!(AckDispatchStats, unresolved_flushes),
            ),
            (
                "timestamps_enabled",
                std::mem::offset_of!(AckDispatchStats, timestamps_enabled),
            ),
        ] {
            let expected = match name {
                "size" => 0,
                "version" => 4,
                "items" => 8,
                "commands" => 16,
                "barriers" => 24,
                "flushes" => 32,
                "timed_flushes" => 40,
                "timed_items" => 48,
                "device_ns" => 56,
                "min_item_ns" => 64,
                "max_item_ns" => 72,
                "unresolved_flushes" => 80,
                "timestamps_enabled" => 88,
                _ => unreachable!(),
            };
            assert_eq!(offset, expected, "{name}");
        }
    }

    #[test]
    fn the_dispatch_stats_entry_refuses_a_null_out_and_a_short_struct() {
        // The struct's own size IS the handshake for this entry, because the
        // ABI version did not move for it: a caller whose mirror is smaller
        // than this library's must be refused rather than handed a prefix.
        let mut stats: AckDispatchStats = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe {
                ack_dispatch_stats(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::mem::size_of::<AckDispatchStats>(),
                )
            },
            ACK_ERR_NULL,
            "the out pointer is checked before the device"
        );
        assert_eq!(
            unsafe {
                ack_dispatch_stats(
                    std::ptr::null(),
                    &mut stats,
                    std::mem::size_of::<AckDispatchStats>() - 1,
                )
            },
            ACK_ERR_ABI,
            "a short struct is refused before the device is touched"
        );
        // and only then does it look at the device, so a bad handle cannot be
        // mistaken for a layout mismatch
        assert_eq!(
            unsafe {
                ack_dispatch_stats(
                    std::ptr::null(),
                    &mut stats,
                    std::mem::size_of::<AckDispatchStats>(),
                )
            },
            ACK_ERR_NULL,
            "a null device handle is refused by `device`, after the layout check"
        );
    }

    #[test]
    fn the_identity_changes_with_any_numeric_field_and_not_with_the_names() {
        let mut info: AckDeviceInfo = unsafe { std::mem::zeroed() };
        let base = info_identity(&info);
        info.features |= ACK_FEATURE_COOPMAT_BF16_16X16X16_SUBGROUP;
        let with_feature = info_identity(&info);
        assert_ne!(base, with_feature, "a feature bit is part of the identity");
        info.max_push_constant_bytes = 256;
        assert_ne!(
            with_feature,
            info_identity(&info),
            "a limit is part of the identity"
        );
        info.driver_version_raw = 7;
        let with_driver = info_identity(&info);
        assert_ne!(with_feature, with_driver);
        fill_c_string(&mut info.device_name, "another name");
        assert_eq!(
            with_driver,
            info_identity(&info),
            "names are display text, not identity"
        );
        info.size = 1;
        info.version = 9;
        assert_eq!(
            with_driver,
            info_identity(&info),
            "size and version are outside the hash"
        );
        let mut s = [0 as c_char; 6];
        fill_c_string(&mut s, "ab\u{e9}cd"); // e-acute is two bytes; 5 usable bytes cut after it
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(s.as_ptr()) }
                .to_str()
                .unwrap(),
            "ab\u{e9}c"
        );
        let mut t = [0 as c_char; 4];
        fill_c_string(&mut t, "ab\u{e9}"); // 3 usable bytes: the 2-byte char does not fit, cut before it
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(t.as_ptr()) }
                .to_str()
                .unwrap(),
            "ab"
        );
    }

    #[test]
    fn device_info_refuses_a_closed_handle_and_a_short_struct_without_a_device() {
        let bogus = 0x7ACC_0002usize as *const AckDevice;
        let mut info: AckDeviceInfo = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { ack_device_info(bogus, &mut info, std::mem::size_of::<AckDeviceInfo>()) },
            ACK_ERR_CLOSED
        );
        assert_eq!(
            unsafe { ack_device_info(bogus, &mut info, 16) },
            ACK_ERR_ABI
        );
        assert_eq!(
            unsafe { ack_device_info(bogus, std::ptr::null_mut(), 616) },
            ACK_ERR_NULL
        );
        let mut s = [0 as c_char; 8];
        fill_c_string(&mut s, "0123456789");
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(s.as_ptr()) }
                .to_str()
                .unwrap(),
            "0123456"
        );
    }

    #[test]
    fn shader_identity_is_stable_and_sensitive_to_the_bytes() {
        assert_eq!(shader_identity(), shader_identity());
        assert_ne!(fnv1a(&[b"abc"]), fnv1a(&[b"abd"]));
        // the hash is over the concatenated byte stream: part boundaries do not
        // matter, order does
        assert_eq!(fnv1a(&[b"ab", b"c"]), fnv1a(&[b"abc"]));
        assert_ne!(fnv1a(&[b"abc", b"def"]), fnv1a(&[b"def", b"abc"]));
    }

    #[test]
    fn the_pointwise_operator_has_its_bit_its_schema_and_its_size_refusals() {
        // schema 3 since the bf16 storage switches; the pin is the plan
        // module's own, so the two cannot drift apart
        assert_eq!((ACK_OP_POINTWISE, ACK_OP_SCHEMA_POINTWISE), (8, 3));
        assert_eq!(ACK_OP_SCHEMA_POINTWISE, pointwise_ops::SCHEMA);
        assert_eq!(pointwise_ops::LAST_OP, 23);
        assert_eq!(
            ACK_OP_POINTWISE & (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32),
            0,
            "a fresh operator bit"
        );
        assert_eq!(pointwise_ops::PUSH_BYTES, 120);
        assert_eq!(ACK_ABI_VERSION, 10);
        // a null block and a block of another length refuse before the device
        // lookup (no device is open in this process), and so does an unknown
        // operation word
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let rc = unsafe {
            ack_pointwise(
                dev,
                none,
                none,
                none,
                none,
                std::ptr::null(),
                pointwise_ops::PUSH_BYTES,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ACK_ERR_NULL);
        let short = [0u8; 64];
        let rc = unsafe {
            ack_pointwise(
                dev,
                none,
                none,
                none,
                none,
                short.as_ptr(),
                short.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ACK_ERR_SIZE);
        let mut unknown = [0u8; pointwise_ops::PUSH_BYTES];
        unknown[..4].copy_from_slice(&(pointwise_ops::LAST_OP + 1).to_le_bytes());
        let rc = unsafe {
            ack_pointwise(
                dev,
                none,
                none,
                none,
                none,
                unknown.as_ptr(),
                unknown.len(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ACK_ERR_SHAPE);
        let mut msg = [0 as c_char; 128];
        assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
        let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
        assert!(text.contains("pointwise-operation-out-of-range"), "{text}");
    }

    fn reduce_block(words: &[(usize, u32)]) -> [u8; reduce_ops::PUSH_BYTES] {
        // a sum over a contiguous [4] view, then the caller's overrides
        let mut block = [0u8; reduce_ops::PUSH_BYTES];
        let mut put = |i: usize, w: u32| block[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        put(2, 1); // ndim
        put(3, 1); // reduced mask
        put(4, 4); // shape[0]
        put(10, 1); // stride[0]
        for d in 1..reduce_ops::MAX_RANK {
            put(4 + d, 1);
        }
        for &(i, w) in words {
            put(i, w);
        }
        block
    }

    #[test]
    fn the_reduce_operator_has_its_bit_its_schema_and_its_request_refusals() {
        // schema 2 since 2026-09-10: the request block may carry one storage
        // flags word after the 80-byte push layout, and the 80-byte block
        // still means f32 on both sides
        assert_eq!((ACK_OP_REDUCE, ACK_OP_SCHEMA_REDUCE), (16, 2));
        assert_eq!(
            ACK_OP_REDUCE
                & (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32 | ACK_OP_POINTWISE),
            0,
            "a fresh operator bit"
        );
        assert_eq!(reduce_ops::PUSH_BYTES, 80);
        assert_eq!(reduce_ops::REQUEST_BYTES, 84);
        assert_eq!(reduce_ops::STORAGE_FLAGS, 0b11);
        // a null block and a block of another length refuse before the device
        // lookup (no device is open in this process), and so does every defect
        // of the request itself
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let call = |block: *const u8, len: usize| unsafe {
            ack_reduce(dev, none, none, none, block, len, std::ptr::null_mut())
        };
        assert_eq!(call(std::ptr::null(), reduce_ops::PUSH_BYTES), ACK_ERR_NULL);
        let short = [0u8; 64];
        assert_eq!(call(short.as_ptr(), short.len()), ACK_ERR_SIZE);
        // 84 is accepted and 82 and 88 are not: the flags word is one word or
        // it is absent, and a block between the two lengths is a caller error
        // rather than a partially-read flags word
        let between = [0u8; 82];
        assert_eq!(call(between.as_ptr(), between.len()), ACK_ERR_SIZE);
        let over = [0u8; 88];
        assert_eq!(call(over.as_ptr(), over.len()), ACK_ERR_SIZE);
        // a bit outside STORAGE_FLAGS is refused BY NAME before the device
        // lookup, exactly as every other request defect below is
        let mut flagged = [0u8; reduce_ops::REQUEST_BYTES];
        flagged[..reduce_ops::PUSH_BYTES].copy_from_slice(&reduce_block(&[]));
        flagged[80] = 0b100;
        assert_eq!(call(flagged.as_ptr(), flagged.len()), ACK_ERR_SHAPE);
        let mut msg = [0 as c_char; 128];
        assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
        let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
        assert!(text.contains("reduce-storage-flags-out-of-range"), "{text}");
        // and each of the four ADMITTED words reaches the null device handle
        // instead, which is what makes the refusal above about the flags word
        // rather than about the longer block
        for word in [0u32, 0b01, 0b10, 0b11] {
            let mut ok = [0u8; reduce_ops::REQUEST_BYTES];
            ok[..reduce_ops::PUSH_BYTES].copy_from_slice(&reduce_block(&[]));
            ok[80..].copy_from_slice(&word.to_le_bytes());
            assert_eq!(
                call(ok.as_ptr(), ok.len()),
                ACK_ERR_NULL,
                "storage word {word:#b} is admitted by the request decoder"
            );
        }
        for (words, name) in [
            (&[(0usize, 3u32)][..], "reduce-operation-out-of-range"),
            (&[(1, 3)][..], "reduce-finish-out-of-range"),
            (&[(2, 7)][..], "reduce-rank-out-of-range"),
            (&[(2, 0)][..], "reduce-rank-out-of-range"),
            (&[(17, 2)][..], "reduce-reserved-word-nonzero"),
            (&[(18, 65_536)][..], "reduce-reserved-word-nonzero"),
            // two broadcast reduced dimensions of 2^17 each: 2^34 reduced
            // elements addressed by a uint, with every address still tiny
            (
                &[
                    (2, 3u32),
                    (3, 0b110),
                    (5, 1 << 17),
                    (6, 1 << 17),
                    (11, 0),
                    (12, 0),
                ][..],
                "reduce-reduced-count-above-u32",
            ),
            (&[(5, 2)][..], "reduce-padding-out-of-range"),
            (&[(11, 1)][..], "reduce-padding-out-of-range"),
        ] {
            let block = reduce_block(words);
            assert_eq!(call(block.as_ptr(), block.len()), ACK_ERR_SHAPE, "{name}");
            let mut msg = [0 as c_char; 128];
            assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
            let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
            assert!(text.contains(name), "{text}");
        }
        // a well-formed request meets the null device handle next
        let block = reduce_block(&[]);
        assert_eq!(call(block.as_ptr(), block.len()), ACK_ERR_NULL);
    }

    #[test]
    fn the_rowwise_operator_has_its_bit_its_schema_and_its_request_refusals() {
        assert_eq!((ACK_OP_ROWWISE, ACK_OP_SCHEMA_ROWWISE), (32, 1));
        assert_eq!(
            ACK_OP_ROWWISE
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE),
            0,
            "a fresh operator bit"
        );
        assert_eq!(ROWWISE_PUSH_BYTES, 16);
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let call = |block: *const u8, len: usize| unsafe {
            ack_rowwise(
                dev,
                none,
                none,
                none,
                none,
                block,
                len,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(call(std::ptr::null(), ROWWISE_PUSH_BYTES), ACK_ERR_NULL);
        let short = [0u8; 8];
        assert_eq!(call(short.as_ptr(), short.len()), ACK_ERR_SIZE);
        let block = |rows: u32, cols: u32, op: u32, eps: f32| {
            let mut b = [0u8; ROWWISE_PUSH_BYTES];
            for (i, w) in [rows, cols, op, eps.to_bits()].iter().enumerate() {
                b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            b
        };
        for (b, code, name) in [
            (
                block(4, 8, 7, 1e-5),
                ACK_ERR_SHAPE,
                "rowwise-operation-out-of-range",
            ),
            (
                block(0, 8, 0, 1e-5),
                ACK_ERR_SHAPE,
                "rowwise-dimension-out-of-range",
            ),
            // RE-HOMED 2026-09-08: this was the bare literal `1 << 17`, chosen
            // only because it sat above the then-current MAX_COLS of 1<<16. The
            // raise to 1<<18 made 131,072 an ACCEPTED width, so the pin asserted
            // a refusal that no longer happened. Symbol-relative now, like its
            // four siblings in this table, so it tracks the bound instead of
            // restating a number that was true on one day.
            (
                block(4, crate::row_ops::MAX_COLS + 1, 0, 1e-5),
                ACK_ERR_SHAPE,
                "rowwise-dimension-out-of-range",
            ),
            // Still past MAX_ELEMENTS (2^20 * 2^18 = 2^38 against 2^30) and
            // still inside both dimension bounds, so it reaches the element
            // check rather than the dimension one. Widened with MAX_COLS.
            (
                block(1 << 20, crate::row_ops::MAX_COLS, 5, 1e-5),
                ACK_ERR_SIZE,
                "rowwise-element-limit-exceeded",
            ),
            (
                block(4, 8, 2, 0.0),
                ACK_ERR_SHAPE,
                "rowwise-invalid-epsilon",
            ),
        ] {
            assert_eq!(call(b.as_ptr(), b.len()), code, "{name}");
            let mut msg = [0 as c_char; 128];
            assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
            let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
            assert!(text.contains(name), "{text}");
        }
        // a well-formed request meets the null device handle next
        let b = block(4, 8, 0, 1e-5);
        assert_eq!(call(b.as_ptr(), b.len()), ACK_ERR_NULL);
        assert_eq!(rowwise_reads(RowOp::SoftmaxForward), (false, false));
        assert_eq!(rowwise_reads(RowOp::SoftmaxBackward), (true, false));
        assert_eq!(rowwise_reads(RowOp::RmsNormForward), (false, true));
        assert_eq!(rowwise_reads(RowOp::RmsNormBackward), (true, true));
        assert_eq!(rowwise_reads(RowOp::RmsNormGammaReduction), (false, false));
        assert_eq!(rowwise_reads(RowOp::LogSoftmaxForward), (false, false));
        assert_eq!(rowwise_reads(RowOp::LogSoftmaxBackward), (true, false));
    }

    #[test]
    fn the_loss_operator_has_its_bit_its_schema_and_its_plan_refusals() {
        assert_eq!((ACK_OP_LOSS, ACK_OP_SCHEMA_LOSS), (64, 1));
        assert_eq!(
            ACK_OP_LOSS
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE),
            0,
            "a fresh operator bit"
        );
        assert_eq!(LOSS_PUSH_BYTES, 20);
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let targets = [0i64, 1];
        let call = |rows: u32, cols: u32, op: u32, red: u32, ignore: i64, t: &[i64]| unsafe {
            ack_loss(
                dev,
                none,
                none,
                none,
                none,
                rows,
                cols,
                op,
                red,
                ignore,
                t.as_ptr(),
                t.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        // a null target pointer with a nonzero length is refused before anything else
        assert_eq!(
            unsafe {
                ack_loss(
                    dev,
                    none,
                    none,
                    none,
                    none,
                    2,
                    4,
                    3,
                    0,
                    -100,
                    std::ptr::null(),
                    2,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            ACK_ERR_NULL
        );
        // Each case is called INSIDE the loop. An array literal evaluates every
        // call first, leaving ack_last_error holding only the LAST message
        // while the loop checks all seven against it -- which is what the first
        // version of this test did, and it failed for exactly that reason.
        for (rows, cols, o, red, ignore, t, code, name) in [
            (
                2u32,
                4u32,
                5u32,
                0u32,
                -100i64,
                &targets[..],
                ACK_ERR_SHAPE,
                "loss-operation-out-of-range",
            ),
            (
                2,
                4,
                3,
                3,
                -100,
                &targets[..],
                ACK_ERR_SHAPE,
                "loss-reduction-out-of-range",
            ),
            (
                0,
                4,
                3,
                0,
                -100,
                &targets[..],
                ACK_ERR_SHAPE,
                "loss-dimension-out-of-range",
            ),
            (
                2,
                4,
                2,
                0,
                -100,
                &targets[..],
                ACK_ERR_SHAPE,
                "loss-invalid-reduction-for-operation",
            ),
            (
                3,
                4,
                3,
                0,
                -100,
                &targets[..],
                ACK_ERR_SHAPE,
                "loss-target-shape-mismatch",
            ),
            (
                2,
                4,
                3,
                0,
                -100,
                &[0i64, 9][..],
                ACK_ERR_SHAPE,
                "loss-target-out-of-range",
            ),
            (
                2,
                4,
                3,
                2,
                0,
                &[0i64, 0][..],
                ACK_ERR_SHAPE,
                "loss-mean-has-no-valid-targets",
            ),
        ] {
            assert_eq!(call(rows, cols, o, red, ignore, t), code, "{name}");
            let mut msg = [0 as c_char; 160];
            assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 160) }, ACK_OK);
            let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
            assert!(text.contains(name), "{name}: {text}");
        }
        // a well-formed request meets the null device handle next
        assert_eq!(call(2, 4, 3, 0, -100, &targets), ACK_ERR_NULL);
        // The kernel is authoritative for the C ABI's read sets. These are read
        // off kernels/loss_f32.comp, and this is the guard if a future kernel
        // change starts reading a buffer one of these says it does not.
        assert_eq!(loss_reads(LossOp::Forward), (true, false));
        assert_eq!(loss_reads(LossOp::Backward), (true, true));
        assert_eq!(loss_reads(LossOp::Reduce), (true, false));
        assert_eq!(loss_reads(LossOp::NllForward), (true, false));
        assert_eq!(
            loss_reads(LossOp::NllBackward),
            (false, true),
            "the NLL backward reads the target and the upstream gradient, never the logits"
        );
        // and the reported upload is the plan's own image, so a caller that
        // counts transfers has a measurement rather than its own arithmetic
        let plan = LossPlan::new(
            3,
            5,
            LossOp::NllForward,
            LossReduction::None,
            -100,
            &[0, 1, 2],
        )
        .expect("a valid plan");
        assert_eq!(plan.target_bytes().len(), 3 * 4);
    }

    #[test]
    fn the_embedding_operator_has_its_bit_its_schema_and_its_plan_refusals() {
        assert_eq!((ACK_OP_EMBEDDING, ACK_OP_SCHEMA_EMBEDDING), (128, 1));
        assert_eq!(
            ACK_OP_EMBEDDING
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE
                    | ACK_OP_LOSS),
            0,
            "a fresh operator bit"
        );
        assert_eq!(EMBEDDING_PUSH_BYTES, 16);
        assert_eq!(
            ACK_ABI_VERSION, 10,
            "ABI 10 since ack_matmul_bf16_z (2026-09-09); the embedding family joined ABI 9 without bumping it"
        );
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let ids = [0i64, 1];
        let call = |n: u32, vocab: u32, dim: u32, op: u32, t: &[i64]| unsafe {
            ack_embedding(
                dev,
                none,
                none,
                none,
                none,
                n,
                vocab,
                dim,
                op,
                t.as_ptr(),
                t.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            unsafe {
                ack_embedding(
                    dev,
                    none,
                    none,
                    none,
                    none,
                    2,
                    4,
                    3,
                    0,
                    std::ptr::null(),
                    2,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            ACK_ERR_NULL
        );
        for (n, vocab, dim, op, t, name) in [
            (
                2u32,
                4u32,
                3u32,
                2u32,
                &ids[..],
                "embedding-operation-out-of-range",
            ),
            (0, 4, 3, 0, &ids[..], "embedding-dimension-out-of-range"),
            (2, 0, 3, 0, &ids[..], "embedding-dimension-out-of-range"),
            (3, 4, 3, 0, &ids[..], "embedding-index-shape-mismatch"),
            (2, 4, 3, 0, &[0i64, 4][..], "embedding-index-out-of-range"),
            (2, 4, 3, 0, &[0i64, -1][..], "embedding-index-out-of-range"),
        ] {
            assert_eq!(call(n, vocab, dim, op, t), ACK_ERR_SHAPE, "{name}");
            let mut msg = [0 as c_char; 160];
            assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 160) }, ACK_OK);
            let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
            assert!(text.contains(name), "{name}: {text}");
        }
        // a well-formed request meets the null device handle next
        assert_eq!(call(2, 4, 3, 0, &ids), ACK_ERR_NULL);
        // the two images differ by operation, and the entry must not swap them:
        // the gather indexes by id with one unused offset word, the backward by
        // grouped position with vocab + 1 bucket offsets
        let gather =
            EmbeddingPlan::new(2, 4, 3, EmbeddingOp::Gather, &[3, 1]).expect("a valid plan");
        assert_eq!(gather.index_bytes().len(), 2 * 4);
        assert_eq!(gather.offset_bytes().len(), 4);
        let backward =
            EmbeddingPlan::new(2, 4, 3, EmbeddingOp::Backward, &[3, 1]).expect("a valid plan");
        assert_eq!(backward.index_bytes().len(), 2 * 4);
        assert_eq!(backward.offset_bytes().len(), (4 + 1) * 4);
    }

    #[test]
    fn device_ids_are_confined_to_the_owner_half_of_a_buffer_handle() {
        // the registry refuses to hand out an id above u32::MAX, so this
        // encoding never wraps into another device's handles (registry-handles-01)
        let (owner, seq) = split_buffer_handle(buffer_handle(u32::MAX as u64, 1));
        assert_eq!((owner, seq), (u32::MAX as u64, 1));
        assert_ne!(buffer_handle(1, 5), buffer_handle(2, 5));
    }
}
