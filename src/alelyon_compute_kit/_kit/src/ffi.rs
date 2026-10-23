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
use crate::expert_factor_ops::{ExpertFactorOp, ExpertFactorPlan, ExpertFactorPlanError};
use crate::head_loss_ops::{HeadGradPlan, HeadLossPlanError, HeadLsePlan};
use crate::loss_ops::{LossOp, LossPlan, LossPlanError, LossReduction};
use crate::nf4_ops::{
    Nf4DecodePlan, Nf4Elem, Nf4GemvPlan, Nf4MatmulError, Nf4MatmulPlan, Nf4PlanError, Nf4Storage,
    GROUP_WORDS, TABLE_ENTRIES,
};
use crate::reduce_ops::{self, Finish, ReduceOp, ReducePlan, ReducePlanError, ReduceView};
use crate::rotary_ops::{RotaryOp, RotaryPlan};
use crate::row_ops::{RowOp, RowPlan, RowPlanError};
use crate::student_norm_ops::{NormOp, NormPlan, NormPlanError};
use crate::swiglu_ops::{
    ScaledStorage, SwigluOp, SwigluPlan, SwigluPlanError, SwigluScaledPlan, SwigluStorage,
};
use crate::vq4_ops::{VqPlan, VqPlanError};
use crate::{AckError, Buffer, Context, Kernel, Staging, StagingKind, TransferTicket};
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
// The same eight compiled with -DSGW=32 and a 2 x 4 subgroup grid (eight
// subgroups of 32 invocations, each a 64 x 32 tile), for a device whose
// subgroup-size range excludes 64 (NVIDIA's is [32, 32]). A 64-lane module on
// such a device computes the wrong tiles rather than refusing, so the width is
// chosen at open, never inferred.
const SPV_S32_NN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32_128x128_nn.spv");
const SPV_S32_NT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32_128x128_nt.spv");
const SPV_S32_TN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32_128x128_tn.spv");
const SPV_S32_TT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32_128x128_tt.spv");
const SPV_S32_E_NN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32e_128x128_nn.spv");
const SPV_S32_E_NT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32e_128x128_nt.spv");
const SPV_S32_E_TN: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32e_128x128_tn.spv");
const SPV_S32_E_TT: &[u8] = include_bytes!("../kernels/coopmat_bf16_v4s32e_128x128_tt.spv");
// The sixteen above compiled with -DELEM_F16=1: fp16 operands, the f32
// accumulator and output unchanged (ABI 11). For a device whose cooperative
// matrices take fp16 and not bf16, such as NVIDIA Turing. Indexed like the
// bf16 set, edge*4 + a_t*2 + b_t, one array per subgroup width.
const SPV_F16_S64: [&[u8]; 12] = [
    include_bytes!("../kernels/coopmat_f16_v4_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4_128x128_tt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4e_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4e_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4e_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4e_128x128_tt.spv"),
    // the RAST four at RAST_AT (2026-10-01)
    include_bytes!("../kernels/coopmat_f16_v4r_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4r_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4r_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4r_128x128_tt.spv"),
];
const SPV_F16_S32: [&[u8]; 8] = [
    include_bytes!("../kernels/coopmat_f16_v4s32_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32_128x128_tt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32e_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32e_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32e_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32e_128x128_tt.spv"),
];
// nf4-matmul/v1 (2026-09-30, OPTIONAL, resolved by name): the edge modules
// above with -DNF4_B=1, B read as NF4 in its stored form and staged as
// nf4-decode/v1's 16-bit words, the z grid grouped. Indexed [elem][b_t], elem
// 0 bf16 and 1 fp16, one array per subgroup width; each pair built on the first
// `ack_matmul_nf4` call for its element type.
const SPV_NF4_MATMUL_S64: [[&[u8]; 2]; 3] = [
    [
        include_bytes!("../kernels/coopmat_nf4_bf16_v4e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_bf16_v4e_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16_v4e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16_v4e_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16h_v4e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16h_v4e_128x128_nt.spv"),
    ],
];
// nf4-gemv/v1 (2026-09-30, OPTIONAL, resolved by name): the NF4 product at one
// to 16 rows a group, [bf16, fp16] operands, each in modules holding 1, 4 and 16
// rows a group (nf4_ops::GEMV_ROW_CAPS; the same words); each built on the
// first `ack_gemv_nf4` call that needs it. The third row is fp16 with the
// device's own conversions under declared float controls (F16_HW, 2026-09-30;
// the same words wherever no NaN is involved), taken where the device can
// declare them (`gemv_slot`)
const SPV_NF4_GEMV: [[&[u8]; 3]; 3] = [
    [
        include_bytes!("../kernels/nf4_gemv_bf16_r1.spv"),
        include_bytes!("../kernels/nf4_gemv_bf16_r4.spv"),
        include_bytes!("../kernels/nf4_gemv_bf16.spv"),
    ],
    [
        include_bytes!("../kernels/nf4_gemv_f16_r1.spv"),
        include_bytes!("../kernels/nf4_gemv_f16_r4.spv"),
        include_bytes!("../kernels/nf4_gemv_f16.spv"),
    ],
    [
        include_bytes!("../kernels/nf4_gemv_f16h_r1.spv"),
        include_bytes!("../kernels/nf4_gemv_f16h_r4.spv"),
        include_bytes!("../kernels/nf4_gemv_f16h.spv"),
    ],
];
const SPV_NF4_MATMUL_S32: [[&[u8]; 2]; 3] = [
    [
        include_bytes!("../kernels/coopmat_nf4_bf16_v4s32e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_bf16_v4s32e_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16_v4s32e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16_v4s32e_128x128_nt.spv"),
    ],
    // fp16 with the device's own conversion of each decoded weight under
    // declared float controls (NF4_HW, 2026-09-30; the same words wherever no
    // NaN is involved), taken where the device can declare them (`nf4_matmul_slot`)
    [
        include_bytes!("../kernels/coopmat_nf4_f16h_v4s32e_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16h_v4s32e_128x128_nt.spv"),
    ],
];
// nf4-matmul/v1 with a 16-bit C (2026-10-01, OPTIONAL, resolved by name): each
// pair above with -DBF16_OUT=1, so the product's own epilogue writes C in the
// operands' type -- bf16 rounded as cast_f32_bf16 rounds, fp16 by the device's
// own conversion under declared float controls -- for `ack_matmul_nf4_to_16`.
// Indexed like `SPV_NF4_MATMUL_S64`, [nf4_matmul_slot][b_t]; each pair built on
// the first call that needs it.
const SPV_NF4_MATMUL_OUT16_S64: [[&[u8]; 2]; 3] = [
    [
        include_bytes!("../kernels/coopmat_nf4_bf16_v4eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_bf16_v4eo_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16_v4eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16_v4eo_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16h_v4eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16h_v4eo_128x128_nt.spv"),
    ],
];
const SPV_NF4_MATMUL_OUT16_S32: [[&[u8]; 2]; 3] = [
    [
        include_bytes!("../kernels/coopmat_nf4_bf16_v4s32eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_bf16_v4s32eo_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16_v4s32eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16_v4s32eo_128x128_nt.spv"),
    ],
    [
        include_bytes!("../kernels/coopmat_nf4_f16h_v4s32eo_128x128_nn.spv"),
        include_bytes!("../kernels/coopmat_nf4_f16h_v4s32eo_128x128_nt.spv"),
    ],
];
// The four edge layouts compiled with -DBF16_OUT=1 (2026-09-27): C written as
// bf16 in the product's own epilogue, rounded as cast_f32_bf16 rounds, for
// the OPTIONAL `ack_matmul_bf16_to_bf16`. Indexed a_t*2 + b_t, one array per
// subgroup width.
const SPV_BF16_OUT_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4eo_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4eo_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4eo_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4eo_128x128_tt.spv"),
];
const SPV_BF16_OUT_S32: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4s32eo_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32eo_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32eo_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32eo_128x128_tt.spv"),
];
// The same epilogue over the ALIGNED staging (2026-09-28), for a shape on the
// 128 x 128 x 32 tile (`bf16_out_aligned`). The edge gather costs even when every
// tile is inside the extent: an RX 9070 XT ran 1024 x 32000 x 512 nt in 0.431 ms
// against the edge module's 0.515, and 1024 x 512 x 32000 nn in 0.632 against
// 1.322 (single dispatches, src/bin/gemm_bench.rs). The staging and the product
// sequence are the aligned modules', so C is the same bits. Indexed like the
// edge sets, a_t*2 + b_t.
const SPV_BF16_OUT_ALIGNED_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4o_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4o_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4o_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4o_128x128_tt.spv"),
];
const SPV_BF16_OUT_ALIGNED_S32: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4s32o_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32o_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32o_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4s32o_128x128_tt.spv"),
];

/// Whether `ack_matmul_bf16_to_bf16` may take the aligned bf16-output module:
/// every extent on the 128 x 128 x 32 tile, the rule `ack_matmul_bf16` takes its
/// own aligned modules by, so the two products pair aligned with aligned. Its operands are the entry's own
/// dense ones (the logical widths are the pitches), so the pitch rule follows.
fn bf16_out_aligned(m: u32, n: u32, k: u32) -> bool {
    m % TILE == 0 && n % TILE == 0 && k % K_MULTIPLE == 0
}
// The fp16 twins of the four bf16-output sets (2026-09-30): each set's flags
// with -DELEM_F16=1, so C is written as fp16 in the product's own epilogue,
// rounded by convert/v1's f32 -> f16 -- bit for bit what `ack_matmul_f16`
// followed by convert/v1 writes -- for the OPTIONAL `ack_matmul_f16_to_f16`.
// Indexed like them, a_t*2 + b_t; the aligned sets for a shape on the
// 128 x 128 x 32 tile (`bf16_out_aligned`, the rule `ack_matmul_f16` takes its
// own aligned modules by).
const SPV_F16_OUT_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_f16_v4eo_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4eo_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4eo_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4eo_128x128_tt.spv"),
];
const SPV_F16_OUT_S32: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_f16_v4s32eo_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32eo_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32eo_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32eo_128x128_tt.spv"),
];
const SPV_F16_OUT_ALIGNED_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_f16_v4o_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4o_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4o_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4o_128x128_tt.spv"),
];
const SPV_F16_OUT_ALIGNED_S32: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_f16_v4s32o_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32o_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32o_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4s32o_128x128_tt.spv"),
];
// The GROUPED WORKGROUP ORDER (2026-10-01, the shader's WORKGROUP ORDER note):
// the four 64-lane aligned sets compiled with -DRAST=4, so consecutive
// workgroups walk four tile rows down one tile column before the next and a
// panel of B is reused while it is cached. Only the map from workgroup to tile
// changes, so C is the row-order module's, bit for bit. Each 64-lane set holds
// its four at `RAST_AT` (after its eight others; the fp16 f32-output four are in
// SPV_F16_S64 itself), and a product takes them only through `rast_index`.
// Indexed a_t*2 + b_t. The 32-lane sets have none: unmeasured on their devices.
const SPV_RAST_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4r_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4r_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4r_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4r_128x128_tt.spv"),
];
const SPV_BF16_OUT_RAST_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_bf16_v4ro_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4ro_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4ro_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_bf16_v4ro_128x128_tt.spv"),
];
const SPV_F16_OUT_RAST_S64: [&[u8]; 4] = [
    include_bytes!("../kernels/coopmat_f16_v4ro_128x128_nn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4ro_128x128_nt.spv"),
    include_bytes!("../kernels/coopmat_f16_v4ro_128x128_tn.spv"),
    include_bytes!("../kernels/coopmat_f16_v4ro_128x128_tt.spv"),
];
/// The narrowest grid, in tile columns, that takes the grouped order. MEASURED
/// on an RX 9070 XT, device time, bf16, RAST=4 against the row-order module
/// (src/bin/gemm_bench.rs; the shader's WORKGROUP ORDER note): at 64 tile
/// columns or more, f32 C 1.22-1.94x in every layout and bf16 C 1.08-1.36x
/// (8192^3, 2048 x 32768 x 1024); at 32 or fewer 0.91-1.02x (4096^3, 8192 x
/// 2048 x 768, 32000 x 512 x 1024, 2048 x 2048 x 8192); at 48 (8192 x 6144 x
/// 2048 nt) 1.01x in one order and 1.15x in the other, which is no rule, so
/// the row order keeps every grid below 64.
const RAST_MIN_COLS: u32 = 64;
/// Where a 64-lane product set holds its four RAST modules.
const RAST_AT: usize = 8;

/// The index of the RAST module a product takes, or `None` for the module it
/// takes by its set's own rule: an ALIGNED product on a grid at least
/// `RAST_MIN_COLS` tile columns wide, on a set built with its RAST modules.
fn rast_index(aligned: bool, n: u32, set_len: usize, a_t: i32, b_t: i32) -> Option<usize> {
    (aligned && set_len >= RAST_AT + 4 && n.div_ceil(TILE) >= RAST_MIN_COLS)
        .then(|| RAST_AT + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize))
}
const SPV_CAST_F32_BF16: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");
const SPV_CAST_BF16_F32: &[u8] = include_bytes!("../kernels/cast_bf16_f32.spv");
const SPV_MATMUL_F32: &[u8] = include_bytes!("../kernels/matmul_f32.spv");
const SPV_MATMUL_F32_V2: &[u8] = include_bytes!("../kernels/matmul_f32_v2.spv");
const SPV_POINTWISE: &[u8] = include_bytes!("../kernels/pointwise_f32.spv");
// The same family over bf16 storage (schema 3), one switch per buffer, then
// over bool storage (schema 4, 2026-09-22): indexed as
// `pointwise_ops::storage_module` - 1. The first fifteen are the bf16 bits,
// x, y, z, out from the low end; the rest are `pointwise_ops::BOOL_MODULES`
// in order, named by each slot's storage (f f32, b bf16, u bool), then over
// binary16 storage (schema 7, 2026-09-29): `pointwise_ops::F16_MODULES`, named
// by their binary16 slots as the bf16 ones are, and
// `pointwise_ops::F16_BOOL_MODULES` (h binary16). Index 0 of the module
// numbering is SPV_POINTWISE above.
const SPV_POINTWISE_STORAGE: [&[u8]; pointwise_ops::STORAGE_MODULES - 1] = [
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
    include_bytes!("../kernels/pointwise_bool_fffu.spv"),
    include_bytes!("../kernels/pointwise_bool_bbbu.spv"),
    include_bytes!("../kernels/pointwise_bool_fbfu.spv"),
    include_bytes!("../kernels/pointwise_bool_bfbu.spv"),
    include_bytes!("../kernels/pointwise_bool_uuuu.spv"),
    include_bytes!("../kernels/pointwise_bool_uuuf.spv"),
    include_bytes!("../kernels/pointwise_bool_uuub.spv"),
    include_bytes!("../kernels/pointwise_bool_ufff.spv"),
    include_bytes!("../kernels/pointwise_bool_uffb.spv"),
    include_bytes!("../kernels/pointwise_bool_ufbf.spv"),
    include_bytes!("../kernels/pointwise_bool_ufbb.spv"),
    include_bytes!("../kernels/pointwise_bool_ubff.spv"),
    include_bytes!("../kernels/pointwise_bool_ubfb.spv"),
    include_bytes!("../kernels/pointwise_bool_ubbf.spv"),
    include_bytes!("../kernels/pointwise_bool_ubbb.spv"),
    include_bytes!("../kernels/pointwise_f16_xyzout.spv"),
    include_bytes!("../kernels/pointwise_f16_y.spv"),
    include_bytes!("../kernels/pointwise_f16_xz.spv"),
    include_bytes!("../kernels/pointwise_bool_hhhu.spv"),
    include_bytes!("../kernels/pointwise_bool_fhfu.spv"),
    include_bytes!("../kernels/pointwise_bool_hfhu.spv"),
    include_bytes!("../kernels/pointwise_bool_uhhh.spv"),
    include_bytes!("../kernels/pointwise_bool_uhff.spv"),
    include_bytes!("../kernels/pointwise_bool_ufhf.spv"),
];
/// The pointwise family's four-element VECTOR modules (2026-09-27), indexed
/// by `pointwise_ops::vector_module`: 0 is f32, k below 16 is bf16 storage
/// module k, and 16 onwards are `pointwise_ops::F16_MODULES`' in order (schema
/// 7). `pointwise_ops::PointwisePlan::vector_eligible` picks one; bool storage
/// has none.
const SPV_POINTWISE_VECTOR: [&[u8]; pointwise_ops::VECTOR_MODULES] = [
    include_bytes!("../kernels/pointwise_vec_f32.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_x.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_y.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xy.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_z.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xz.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_yz.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xyz.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_out.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_yout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xyout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_zout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xzout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_yzout.spv"),
    include_bytes!("../kernels/pointwise_vec_bf16_xyzout.spv"),
    include_bytes!("../kernels/pointwise_vec_f16_xyzout.spv"),
    include_bytes!("../kernels/pointwise_vec_f16_y.spv"),
    include_bytes!("../kernels/pointwise_vec_f16_xz.spv"),
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
/// reduce-f32/v3 (2026-09-26): rows of a contiguous reduced span own a few
/// lanes each. Indexed like a pass's storage module: 0 f32/f32, then bf16
/// in, bf16 out, both. Built on the first reduction, not at open: the f32
/// module then, the three bf16 ones with the rest of the bf16 set.
const SPV_REDUCE_V3: [&[u8]; 4] = [
    include_bytes!("../kernels/reduce_f32_v3.spv"),
    include_bytes!("../kernels/reduce_v3_bf16_in.spv"),
    include_bytes!("../kernels/reduce_v3_bf16_out.spv"),
    include_bytes!("../kernels/reduce_v3_bf16_inout.spv"),
];
const SPV_ROWWISE: &[u8] = include_bytes!("../kernels/rowwise_f32.spv");
const SPV_VQ: &[u8] = include_bytes!("../kernels/vq4.spv");
pub const VQ_PUSH_BYTES: usize = crate::vq4_ops::PUSH_BYTES;
const SPV_LOSS: &[u8] = include_bytes!("../kernels/loss_f32.spv");
// The same source over bf16 storage (2026-09-28), for the optional
// `ack_loss_bf16`: [the logits read as bf16, and the gradient written as bf16 too].
const SPV_LOSS_BF16: [&[u8]; 2] = [
    include_bytes!("../kernels/loss_bf16_in.spv"),
    include_bytes!("../kernels/loss_bf16_inout.spv"),
];
/// `ack_loss_bf16`'s storage flags: the primary (the logits) holds bf16, for
/// the forward and the backward.
pub const ACK_LOSS_PRIMARY_BF16: u32 = 1;
/// The backward's output (the gradient) holds bf16, each value rounded to
/// nearest even as the cast kernel rounds it. Needs `ACK_LOSS_PRIMARY_BF16`.
pub const ACK_LOSS_OUTPUT_BF16: u32 = 2;
// rotary-f32/v1 (2026-09-29, OPTIONAL, resolved by name): the rotate-half
// rotary position embedding and its backward; built on the first `ack_rotary` call
const SPV_ROTARY: &[u8] = include_bytes!("../kernels/rotary_f32.spv");
// head-kl-ce-f16/v1 (2026-09-29, OPTIONAL, resolved by name): the sub-bit student's head
// loss over an fp16 head output, its forward's two log-sum-exps and its backward; built on
// the first `ack_head_lse_f16` / `ack_head_grad_f16` call
const SPV_HEAD_LSE: &[u8] = include_bytes!("../kernels/head_lse_f16.spv");
const SPV_HEAD_GRAD: &[u8] = include_bytes!("../kernels/head_grad_f16.spv");
// expert-factor-f16/v1 (2026-09-29, OPTIONAL, resolved by name): the sub-bit student's expert
// factors from their signs, and their backward; built on the first `ack_expert_factor_f16` call
const SPV_EXPERT_FACTOR: &[u8] = include_bytes!("../kernels/expert_factor_f16.spv");
// swiglu/v1 (2026-09-29, OPTIONAL, resolved by name): the SiLU-gated activation
// of one packed projection and its backward, [FP32, bf16] storage; each module
// built on the first `ack_swiglu` call that asks for it
// weighted-bag/v1 (2026-10-22, OPTIONAL, resolved by name): the weighted sum of gathered table rows and its two
// gradients, FP32; built on the first `ack_weighted_bag` call
const SPV_WEIGHTED_BAG: &[u8] = include_bytes!("../kernels/weighted_bag.spv");
const SPV_SWIGLU: [&[u8]; 2] = [
    include_bytes!("../kernels/swiglu_f32.spv"),
    include_bytes!("../kernels/swiglu_bf16.spv"),
];
// swiglu-scaled/v1 (2026-09-30, OPTIONAL, resolved by name): the sub-bit student's
// experts' activation silu(gate) * (up * g) and its backward, [FP32, bf16, fp16]
// storage; each module built on the first `ack_swiglu_scaled` call that asks for it
const SPV_SWIGLU_SCALED: [&[u8]; 3] = [
    include_bytes!("../kernels/swiglu_scaled_f32.spv"),
    include_bytes!("../kernels/swiglu_scaled_bf16.spv"),
    include_bytes!("../kernels/swiglu_scaled_f16.spv"),
];
// student-norm-f16/v1 (2026-09-29, OPTIONAL, resolved by name): the sub-bit student's attention-block
// norms and q/k norms with RoPE; built on the first `ack_student_norm_f16` call
const SPV_STUDENT_NORM: &[u8] = include_bytes!("../kernels/student_norm_f16.spv");
// nf4-decode/v1 (2026-09-30, OPTIONAL, resolved by name): bitsandbytes' NF4
// decode into [FP32, bf16, fp16]; each module built on the first
// `ack_nf4_decode` call that asks for it
const SPV_NF4_DECODE: [&[u8]; 3] = [
    include_bytes!("../kernels/nf4_decode_f32.spv"),
    include_bytes!("../kernels/nf4_decode_bf16.spv"),
    include_bytes!("../kernels/nf4_decode_f16.spv"),
];
const SPV_EMBEDDING: &[u8] = include_bytes!("../kernels/embedding_f32.spv");
// index/v1 (ABI 12, 2026-09-26): gather, scatter and scatter-add over strided
// views of 1-, 2-, 4- and 8-byte elements; built on the first `ack_index` call
const SPV_INDEX: &[u8] = include_bytes!("../kernels/index.spv");
// convert/v1 (ABI 13, 2026-09-27): a dtype conversion over strided views,
// between f32, bf16, f16, the integer types, f64 and bool; built on the first
// `ack_convert` call
const SPV_CONVERT: &[u8] = include_bytes!("../kernels/convert.spv");
// and its tiled transposed-read module (2026-10-03, convert_ops::FLAG_TILED),
// built on the first plan that takes it
const SPV_CONVERT_TILED: &[u8] = include_bytes!("../kernels/convert_tiled.spv");
const SPV_INTWISE: &[u8] = include_bytes!("../kernels/intwise.spv");
// intwise/v1 schema 2 (2026-09-28): the binary64 module, the same source built
// with -DF64=1, for C = f64 (torch's float64 arithmetic in integer words);
// built on the first `ack_intwise` call that asks for it
const SPV_INTWISE_F64: &[u8] = include_bytes!("../kernels/intwise_f64.spv");
// radix/v1 (intwise schema 6, 2026-09-29): the stable ranks of long rows by a
// radix sort; built on the first `ack_intwise` call whose rank plans it
const SPV_RADIX: &[u8] = include_bytes!("../kernels/radix.spv");
// attention-v2 (2026-09-27): forward, dQ and dK/dV, one module per head
// dimension and subgroup width, [D 64, D 128] x [64 lanes, 32 lanes]. They
// replace attention-bf16/v1, whose six modules declared 40-41 KiB of shared
// memory against the 32 KiB a driver may report, and which ran 2.4-3.5x slower
// on an RX 9070 XT (tests/attention_v2.rs times both).
const SPV_ATTN_FWD: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_fwd_v2_d64_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_fwd_v2_d128_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_d128_s32.spv"),
    ],
];
const SPV_ATTN_DQ: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_d128_s32.spv"),
    ],
];
const SPV_ATTN_DKDV: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_d128_s32.spv"),
    ],
];
// attention-f16 (2026-09-29): the same three kernels built over binary16
// (-DELT_F16=1), the dtype stage E's fp16 autocast hands SDPA; the bf16 modules
// above are byte for byte what they were before the switch
const SPV_ATTN_F16_FWD: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_fwd_v2_f16_d64_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_f16_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_fwd_v2_f16_d128_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_f16_d128_s32.spv"),
    ],
];
const SPV_ATTN_F16_DQ: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_f16_d128_s32.spv"),
    ],
];
const SPV_ATTN_F16_DKDV: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16_d128_s32.spv"),
    ],
];
// attention-f16 with the device's own binary16 conversions (2026-09-29): the
// same sources with -DF16_HW=1, which declare round to nearest even and
// denormal preservation for 16-bit floats. Built only for a device that
// reports both (`attention_f16_hardware`); bit for bit the integer modules
// above, which stay the fallback and the measured control.
const SPV_ATTN_F16HW_FWD: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d64_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d128_s64.spv"),
        include_bytes!("../kernels/attention_fwd_v2_f16hw_d128_s32.spv"),
    ],
];
const SPV_ATTN_F16HW_DQ: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dq_v2_f16hw_d128_s32.spv"),
    ],
];
const SPV_ATTN_F16HW_DKDV: [[&[u8]; 2]; 2] = [
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d64_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d64_s32.spv"),
    ],
    [
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d128_s64.spv"),
        include_bytes!("../kernels/attention_bwd_dkdv_v2_f16hw_d128_s32.spv"),
    ],
];
/// The fused attention's element types, the first index of `Inner::attention`.
const ATTN_BF16: usize = 0;
const ATTN_F16: usize = 1;
const ATTN_ELT_NAMES: [&str; 2] = ["bf16", "f16"];
/// [element type][forward, dQ, dK/dV][head dimension slot][subgroup width slot]
const ATTN_SPV: [[[[&[u8]; 2]; 2]; 3]; 2] = [
    [SPV_ATTN_FWD, SPV_ATTN_DQ, SPV_ATTN_DKDV],
    [SPV_ATTN_F16_FWD, SPV_ATTN_F16_DQ, SPV_ATTN_F16_DKDV],
];
/// `ATTN_SPV[ATTN_F16]`'s hardware-conversion twin.
const ATTN_SPV_F16HW: [[[&[u8]; 2]; 2]; 3] =
    [SPV_ATTN_F16HW_FWD, SPV_ATTN_F16HW_DQ, SPV_ATTN_F16HW_DKDV];

/// Whether the fp16 attention takes the device's own binary16 conversions. It
/// does when the device reports shaderFloat16, shaderRoundingModeRTEFloat16 and
/// shaderDenormPreserveFloat16, and `ACK_F16_HW_CONVERT` is not off. The
/// integer modules are the fallback, and with `ACK_F16_HW_CONVERT=0` the
/// measured control. MEASURED before the switch (RX 9070 XT, GPU timestamps,
/// the AKV shape): the integer rounding made fp16's dK/dV 62% slower than
/// bf16's (0.369 ms against 0.228).
fn attention_f16_hardware(ctx: &Context) -> bool {
    let r = &ctx.report;
    r.shader_float16
        && r.f16_rounding_rte
        && r.f16_denorm_preserve
        && crate::batch::env_flag("ACK_F16_HW_CONVERT", true)
}

use crate::amp_ops::{self, AmpPlan, AmpUnscaleRequest, AmpUpdatePlan, AmpUpdateRequest};
use crate::attention_fused;
use crate::convert_ops::{self, ConvertPlan};
use crate::index_ops::{self, IndexPlan};
use crate::intwise_ops;
use crate::matmul_ops;
use crate::pointwise_ops;
use crate::radix_ops;
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
/// 11: `ack_matmul_f16` and `ack_matmul_f16_z`, the coopmat product with fp16
/// operands (operator bit 256, schema 1: both entries, the z grid included),
/// were added with `AckAbiInfo::op_schema_matmul_f16` and a zero `reserved`
/// word after it, so the struct grew from 80 to 88 bytes with no implicit
/// padding; the device record's feature bits gained
/// `ACK_FEATURE_SHADER_FLOAT16` and `ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP`.
/// The bf16 entries keep their signatures, their modules and their results
/// (2026-09-26).
/// 12: `ack_index`, the index family (index/v1: gather, scatter and
/// scatter-add over strided views of 1-, 2-, 4- and 8-byte elements; operator
/// bit 512, schema 1), was added, and ABI 11's `reserved` word became
/// `AckAbiInfo::op_schema_index`, so the struct keeps its 88 bytes and every
/// field its offset (2026-09-26).
/// 13: `ack_convert`, the conversion family (convert/v1: a dtype conversion
/// over strided views between f32, bf16, f16, i64, i32, i16, i8, u8, f64 and
/// bool; operator bit 1024, schema 1), was added with
/// `AckAbiInfo::op_schema_convert` and a zero `reserved` word after it, so the
/// struct grew from 88 to 96 bytes with no implicit padding (2026-09-27).
/// 14: `ack_intwise`, the integer pointwise family (intwise/v1: add, sub, mul,
/// the bitwise operations and shifts, the comparisons, neg, abs, fill and
/// arange on i64, i32, i16, i8, u8 and bool, computed as torch's CPU kernels
/// compute them; operator bit 2048, schema 1), was added, and ABI 13's
/// `reserved` word became `AckAbiInfo::op_schema_intwise`, so the struct keeps
/// its 96 bytes and every field its offset (2026-09-28).
pub const ACK_ABI_VERSION: u32 = 14;
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
/// an operation word it cannot name. Schema 6 (2026-09-28) carries 0-42: tanh,
/// log and xlogy, in correctly rounded adds and multiplies only.
pub const ACK_OP_POINTWISE: u64 = 8;
pub const ACK_OP_SCHEMA_POINTWISE: u32 = pointwise_ops::SCHEMA;
/// The fixed-order reduction family: `ack_reduce` with the 80-byte request
/// block (reduce-f32/v1, ABI 7).
pub const ACK_OP_REDUCE: u64 = 16;
/// The reduce family's float64 module (schema 4, 2026-09-29): v1's plan over
/// float64 words, a sum with no finish, the binary64 add in integer
/// instructions (`kernels/reduce_f64.comp`). Built on the first reduction that
/// sets `reduce_ops::FLAG_F64`.
const SPV_REDUCE_F64: &[u8] = include_bytes!("../kernels/reduce_f64.spv");
/// amp-unscale-f32/v1 (2026-09-29): GradScaler's non-finite check and unscale
/// of one float32 gradient (`kernels/amp_unscale_f32.comp`, three storage
/// buffers, a 16-byte push block). Built on the first `ack_amp_unscale` call.
const SPV_AMP_UNSCALE: &[u8] = include_bytes!("../kernels/amp_unscale_f32.spv");
/// amp-update-scale-f32/v1 (2026-09-29): GradScaler's scale update in one
/// invocation (`kernels/amp_update_scale_f32.comp`, three storage buffers, a
/// 24-byte push block). Built on the first `ack_amp_update_scale` call.
const SPV_AMP_UPDATE: &[u8] = include_bytes!("../kernels/amp_update_scale_f32.spv");
/// 2 since 2026-09-10: `ack_reduce` also accepts a request block of
/// `reduce_ops::REQUEST_BYTES` -- the same 80-byte push layout followed by
/// one STORAGE FLAGS word (bit 0 the input buffer holds bf16, bit 1 the
/// output does). The 80-byte block still means f32 on both sides and is
/// bit-for-bit the call it always was, no signature or struct changed, and
/// the kernel's own push block is untouched: the flags word never reaches
/// a shader, it selects which shader.
/// 3 since 2026-09-27: operation 3, `AbsMax` (the largest |x|, the norm of
/// order infinity), in the same block. A kit at schema 2 refuses op 3 as out
/// of range, so a caller that sends it requires 3.
/// 4 since 2026-09-29: storage bit 2, `reduce_ops::FLAG_F64` -- a float64 sum
/// (input, partials and output float64) on v1's plan, `SPV_REDUCE_F64`. A
/// kit at schema 3 refuses the bit by name, so a caller that sends it requires 4.
pub const ACK_OP_SCHEMA_REDUCE: u32 = 4;
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
/// The fp16 cooperative-matrix product: `ack_matmul_f16` and
/// `ack_matmul_f16_z`, the bf16 pair's signatures and semantics with fp16
/// operands (ABI 11). Present in every library; a device without fp16
/// cooperative matrices refuses each call with `ACK_ERR_UNSUPPORTED_OP`.
pub const ACK_OP_MATMUL_F16: u64 = 256;
pub const ACK_OP_SCHEMA_MATMUL_F16: u32 = 1;
/// The index family: `ack_index` with the 112-byte plan block (index/v1,
/// ABI 12): gather, scatter and scatter-add over strided views, with an index
/// out of range reported through a caller-owned status word rather than used;
/// since schema 2, a 1- or 2-byte scatter output in any layout (written with
/// integer atomics where fibers share words, refused before) and a float64
/// scatter-add (IEEE binary64 in integer words, torch's CPU NaN rule; refused
/// before); since schema 3, a long fiber's scatter (elected) or int32/int64
/// scatter-add (integer atomics) an element an invocation.
pub const ACK_OP_INDEX: u64 = 512;
pub const ACK_OP_SCHEMA_INDEX: u32 = index_ops::SCHEMA;
/// The conversion family: `ack_convert` with the 72-byte plan block
/// (convert/v1, ABI 13): `out = T_out(in)` over strided views, rounded as c10
/// rounds, with a float -> integer conversion refused by name.
pub const ACK_OP_CONVERT: u64 = 1024;
pub const ACK_OP_SCHEMA_CONVERT: u32 = convert_ops::SCHEMA;
/// The integer pointwise family: `ack_intwise` with its 200-byte request
/// (intwise/v1, ABI 14): torch's integer and bool pointwise arithmetic over
/// strided views, in the operation's common dtype, bit for bit; float64
/// arithmetic since schema 2, the integer cumsum since schema 3, tril and triu
/// since schema 4, the stable ranks (sort, topk) since schema 5, and a long
/// row's ranks by a radix sort (radix/v1) since schema 6.
pub const ACK_OP_INTWISE: u64 = 2048;
pub const ACK_OP_SCHEMA_INTWISE: u32 = intwise_ops::SCHEMA;
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
/// Version of `AckDispatchTagTime`.
pub const ACK_DISPATCH_TAG_TIME_VERSION: u32 = 1;
/// Version of `AckRuntimeStats`.
pub const ACK_RUNTIME_STATS_VERSION: u32 = 1;
/// Slots in `AckRuntimeStats::flushes_by_reason`. `flush_reason_count` says
/// how many this library uses; the rest stay zero, so a later reason fits
/// without a new layout.
pub const ACK_FLUSH_REASON_SLOTS: usize = 16;
const _: () = assert!(crate::batch::FLUSH_REASON_COUNT <= ACK_FLUSH_REASON_SLOTS);

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
/// `shaderFloat16` is enabled on the device (ABI 11). The fp16 matmul modules
/// declare the SPIR-V `Float16` capability, so `ack_matmul_f16` needs it.
pub const ACK_FEATURE_SHADER_FLOAT16: u64 = 128;
/// The 16x16x16 fp16 x fp16 -> f32 subgroup-scope cooperative shape the fp16
/// matmul kernels are written for is advertised by the driver (ABI 11).
pub const ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP: u64 = 256;
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
    /// Schema of `ack_matmul_f16` and `ack_matmul_f16_z`; appended in ABI 11.
    pub op_schema_matmul_f16: u32,
    /// Schema of `ack_index` (index/v1). ABI 11 reserved this word as zero for
    /// the next operator family; ABI 12 set it, so no field moved. The family
    /// after this one appends its schema and a new zero word, which keeps the
    /// struct a multiple of 8 bytes with no implicit padding.
    pub op_schema_index: u32,
    /// Schema of `ack_convert` (convert/v1); appended in ABI 13.
    pub op_schema_convert: u32,
    /// Schema of `ack_intwise` (intwise/v1). ABI 13 reserved this word as zero
    /// for the next operator family; ABI 14 set it, so no field moved. The
    /// family after this one appends its schema and a new zero word.
    pub op_schema_intwise: u32,
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

/// One dispatch tag's share of the timestamp instrument
/// (`ack_dispatch_tag_times`, 2026-10-03; `DispatchTagTime` in the crate).
///
/// SELF-DESCRIBING, AS `AckDispatchStats` IS, and for the same reason the ABI
/// version did not move: the entries are found by name, in no pointer table.
/// Every row carries this library's `size` and `version`, which a reader must
/// require to equal its own.
///
/// Summed over a context's rows, `items` is `AckDispatchStats::timed_items`
/// and `device_ns` is `AckDispatchStats::device_ns`: each interval read back
/// is charged to the tag its item was recorded under (0: none was set).
/// `unresolved_items` counts the tag's items whose interval was not read.
#[repr(C)]
pub struct AckDispatchTagTime {
    pub size: u32,
    pub version: u32,
    pub tag: u32,
    pub reserved: u32,
    pub items: u64,
    pub device_ns: u64,
    pub max_item_ns: u64,
    pub unresolved_items: u64,
}

/// What the batch recorder's flushes and the kit's two pools have done
/// (`ack_runtime_stats`, 2026-09-29).
///
/// WHY IT EXISTS. `AckDispatchStats::flushes` counts every submission alike,
/// and they are not alike. A `Readback`, `Upload`, `ExplicitSync`,
/// `TimedDispatch`, `RetiredBytes` or `Close` flush waits for its batch, so the
/// host stops until the device has drained. A `BatchFull` or
/// `DescriptorsExhausted` flush, with the asynchronous flush on and the
/// timestamps off, submits and returns. Likewise, a buffer the pool cannot
/// serve costs a Vulkan allocation, and one it serves costs a map lookup.
/// `Context` has always counted both (`flushes_by_reason`, `pool_stats`,
/// `staging_pool_stats`), but no caller outside this crate could read them.
///
/// SELF-DESCRIBING, AS `AckDispatchStats` IS. The entry is in no pointer
/// table and is found by name, so the ABI version did not move for it. Its
/// own `size` and `version` are the handshake, and a reader must require both
/// to equal its own. `flushes_by_reason` is in `batch::ALL_FLUSH_REASONS`
/// order, `flush_reason_count` slots of it are used, and
/// `ack_flush_reason_name` names each slot, so a reader never keeps a copy of
/// the list.
///
/// A SNAPSHOT, TAKEN WITHOUT A FLUSH. The pool counters are cumulative for the
/// context; `pool_held_*` and `staging_held_*` are what the pools hold at the
/// call. `batches_in_flight` and `recorded_items` describe the batch state at
/// the call, and the two budgets are the context's current settings.
#[repr(C)]
pub struct AckRuntimeStats {
    pub size: u32,
    pub version: u32,
    /// The slots of `flushes_by_reason` this library uses.
    pub flush_reason_count: u32,
    /// 0 or 1: a submitted batch whose fence the context has not waited for.
    pub batches_in_flight: u32,
    /// Flushes of this context by reason, in `batch::ALL_FLUSH_REASONS` order.
    pub flushes_by_reason: [u64; ACK_FLUSH_REASON_SLOTS],
    /// `Context::buffer` calls served from the pool, and calls that allocated.
    pub pool_hits: u64,
    pub pool_misses: u64,
    /// Freed pairs the pool took back, and pairs it destroyed at its ceiling.
    pub pool_returns: u64,
    pub pool_declined: u64,
    /// Pairs destroyed by a drain, and allocation failures that drained and retried.
    pub pool_drained: u64,
    pub pool_reclaims: u64,
    /// Vulkan objects (two per pair) and allocated bytes the pool holds now.
    pub pool_held_objects: i64,
    pub pool_held_bytes: u64,
    /// The staging pool's counters, as `Context::staging_pool_stats` reports them.
    pub staging_hits: u64,
    pub staging_misses: u64,
    pub staging_returns: u64,
    pub staging_declined: u64,
    pub staging_evicted: u64,
    pub staging_held_pairs: i64,
    pub staging_held_bytes: u64,
    /// The batch depth (`ACK_MAX_RECORDED_ITEMS`), and the items recorded and not yet submitted.
    pub max_recorded_items: u32,
    pub recorded_items: u32,
    /// The retirement budget (`ACK_MAX_RETIRED_BYTES`).
    pub max_retired_bytes: u64,
    /// 1 when a `BatchFull` or `DescriptorsExhausted` flush submits without
    /// waiting (`ACK_ASYNC_FLUSH`, on by default, and the timestamps off);
    /// every other counted flush waits for its batch. 0: every flush waits.
    pub async_flush: u32,
    pub reserved: u32,
}

/// `batch::flush_reason_name` as C strings, in `batch::ALL_FLUSH_REASONS`
/// order. The unit tests hold the two lists equal.
const FLUSH_REASON_C_NAMES: [&std::ffi::CStr; crate::batch::FLUSH_REASON_COUNT] = [
    c"Readback",
    c"Upload",
    c"TimedDispatch",
    c"Close",
    c"ExplicitSync",
    c"BatchFull",
    c"DescriptorsExhausted",
    c"RetiredBytes",
    c"Accounting",
];

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

/// `ACK_COOPMAT_LANES=32` (2026-09-27): open the 32-lane cooperative-matrix
/// sets (the bf16, fp16 and bf16-output products and fused attention) even on
/// a device that admits 64, so a wave64 card can test the modules NVIDIA runs.
/// It is a diagnostic. Unset, or any other value, keeps the rule: 64 first,
/// then 32.
fn narrow_lanes_forced() -> bool {
    std::env::var_os("ACK_COOPMAT_LANES").is_some_and(|v| v == "32")
}

/// `ACK_MATMUL_ORDER=rows` (2026-10-01): build every product set without its
/// RAST modules (the shader's WORKGROUP ORDER note), so every product runs in
/// the row order it always did. It is the measured A/B's control; C is the same
/// bits either way. Read ONCE, at open (`Inner::rows_order`), for every set the
/// device builds then or on its family's first call, so one device's order
/// cannot change with a later change of the environment. Unset, or any other
/// value, keeps the rule (`rast_index`).
fn rows_order_forced() -> bool {
    std::env::var_os("ACK_MATMUL_ORDER").is_some_and(|v| v == "rows")
}

/// `ACK_NF4_GEMV_F16_HW=0` (2026-09-30): keep the fp16 NF4 GEMV on
/// convert/v1's integer modules even on a device that can declare the float
/// controls its device-conversion modules need. It is the measured A/B's
/// control; the words are the same wherever no NaN is involved. Unset, or any
/// other value, keeps the rule.
fn nf4_gemv_f16_hw_refused() -> bool {
    std::env::var_os("ACK_NF4_GEMV_F16_HW").is_some_and(|v| v == "0")
}

/// The row of `SPV_NF4_GEMV` a call takes: bf16's, or for fp16 the modules
/// with the device's own conversions (F16_HW) where the device reports
/// `shaderFloat16` and can declare their float controls (RoundingModeRTE and
/// DenormPreserve, 16- and 64-bit alike), else convert/v1's integer modules.
/// `ACK_NF4_MATMUL_F16_HW=0` (2026-09-30): keep the fp16 NF4 product on the
/// modules that round each decoded weight by convert/v1's integer routine,
/// even on a device that can declare the float controls of the modules that
/// use its own conversion. It is the measured A/B's control; the words are the
/// same wherever no NaN is involved. Unset, or any other value, keeps the rule.
fn nf4_matmul_f16_hw_refused() -> bool {
    std::env::var_os("ACK_NF4_MATMUL_F16_HW").is_some_and(|v| v == "0")
}

/// The row of `SPV_NF4_MATMUL_S64`/`_S32` (and of `Inner::nf4_matmul`) a call
/// takes: bf16's, or for fp16 the modules converting with the device's own
/// instructions (NF4_HW) where the device reports `shaderFloat16` and can
/// declare their float controls, else the integer-rounding modules.
fn nf4_matmul_slot(ctx: &crate::Context, elem: Nf4Elem) -> usize {
    match elem {
        Nf4Elem::Bf16 => 0,
        Nf4Elem::F16
            if ctx.report.shader_float16
                && ctx.fp16_rounding_declarable
                && !nf4_matmul_f16_hw_refused() =>
        {
            2
        }
        Nf4Elem::F16 => 1,
    }
}

fn gemv_slot(ctx: &crate::Context, elem: Nf4Elem) -> usize {
    match elem {
        Nf4Elem::Bf16 => 0,
        Nf4Elem::F16
            if ctx.report.shader_float16
                && ctx.fp16_rounding_declarable
                && !nf4_gemv_f16_hw_refused() =>
        {
            2
        }
        Nf4Elem::F16 => 1,
    }
}

pub(crate) fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = message.into());
}

pub(crate) fn code_for(err: &AckError) -> i32 {
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
pub(crate) unsafe fn no_duration(ms_out: *mut f64) {
    if !ms_out.is_null() {
        unsafe { *ms_out = f64::NAN };
    }
}

/// A `Context::dispatch` result at the C ABI: `ACK_OK` for a recorded
/// dispatch, or the code and message of whatever refused. A refusal that
/// belongs to the deferred batch rather than to this call says so in its
/// message (see `Context::flush`).
pub(crate) fn recorded(r: crate::Result<()>) -> i32 {
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
/// order NN, NT, TN, TT, edge NN, edge NT, edge TN, edge TT, the same eight at
/// subgroup width 32, f32->bf16, bf16->f32, matmul-f32, pointwise-f32,
/// reduce-f32, rowwise-f32, loss-f32, embedding-f32.
///
/// The four edge modules are hashed here because this value exists to tell two
/// libraries of the same version apart by what their kernels compute, and a
/// library that can run an off-tile product is not the one that refused it.
/// Admitting them CHANGES this identity and therefore `build_identity`, so a
/// checkpoint receipt written by an earlier build refuses to resume against
/// this one. That is a declared consequence of the change, not a surprise.
/// The eight width-32 modules are hashed for the same reason and with the
/// same consequence (2026-09-22): a library that computes a correct BF16
/// product on a 32-lane device is not the one that returned a wrong one there.
/// The sixteen fp16 modules (ABI 11: the width-64 eight, then the width-32
/// eight) follow the list above, for the same reason and with the same
/// consequence (2026-09-26): a library that can run an fp16 product is not
/// one that refused it. The index module (ABI 12) follows them, for the same
/// reason and with the same consequence (2026-09-26). The fp16 64-lane set now
/// ends in its four RAST modules, and the bf16 four follow the list
/// (2026-10-01): a library that may run a product in the grouped order is not
/// one that never does, even though C is the same bits. The tiled convert
/// module follows them all (2026-10-03), for the same reason and with the same
/// consequence: its bits are the untiled module's, but a library that may run
/// it is not one that never does.
pub fn shader_identity() -> u64 {
    let mut modules: Vec<&[u8]> = vec![
        SPV_NN,
        SPV_NT,
        SPV_TN,
        SPV_TT,
        SPV_E_NN,
        SPV_E_NT,
        SPV_E_TN,
        SPV_E_TT,
        SPV_S32_NN,
        SPV_S32_NT,
        SPV_S32_TN,
        SPV_S32_TT,
        SPV_S32_E_NN,
        SPV_S32_E_NT,
        SPV_S32_E_TN,
        SPV_S32_E_TT,
        SPV_CAST_F32_BF16,
        SPV_CAST_BF16_F32,
        SPV_MATMUL_F32,
        SPV_MATMUL_F32_V2,
        SPV_POINTWISE,
        SPV_REDUCE,
        SPV_REDUCE_V2,
        SPV_REDUCE_V3[0],
        SPV_ROWWISE,
        SPV_LOSS,
        SPV_EMBEDDING,
        SPV_VQ,
    ];
    modules.extend_from_slice(&SPV_F16_S64);
    modules.extend_from_slice(&SPV_F16_S32);
    modules.push(SPV_INDEX);
    modules.push(SPV_CONVERT);
    modules.push(SPV_INTWISE);
    modules.push(SPV_INTWISE_F64);
    modules.extend_from_slice(&SPV_RAST_S64);
    modules.push(SPV_CONVERT_TILED);
    fnv1a(&modules)
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

/// R4: a staging handle, packed exactly like a buffer handle (its own
/// sequence space, `Inner::next_staging`, so a staging and a buffer handle
/// with the same low 32 bits are still distinguishable by which map is
/// consulted -- `ack_staging_*` entries only ever look in `Inner::stagings`).
pub fn staging_handle(device_id: u64, seq: u32) -> u64 {
    (device_id << 32) | seq as u64
}

/// The (owner, sequence) pair a staging handle encodes.
pub fn split_staging_handle(handle: u64) -> (u64, u32) {
    (handle >> 32, (handle & 0xffff_ffff) as u32)
}

/// Per-device state of a family the public kit does not ship (2026-10-03): its
/// pipelines, built on its entry's first call, kept with the device and handed
/// back to the family's own `destroy` with it, so this file never names the
/// family's types.
pub(crate) type DeviceExtension = (
    Box<dyn std::any::Any + Send>,
    fn(Box<dyn std::any::Any + Send>, &Context),
);

/// What one open device owns. Everything Vulkan sits behind the device's
/// mutex (`AckDevice::inner`), so `Context` may be `Send` and not `Sync` and
/// two threads cannot interleave on one queue (ACK-FFI-01).
pub(crate) struct Inner {
    pub(crate) ctx: Context,
    kernels: Vec<Kernel>, // aligned then edge NN, NT, TN, TT (edge*4 + a_t*2 + b_t); empty without BF16 coopmat
    /// `ACK_MATMUL_ORDER=rows` as it was at open (`rows_order_forced`): every
    /// product set this device builds, at open or on a family's first call,
    /// takes its RAST modules only when this is false.
    rows_order: bool,
    /// The fp16 set (ABI 11), indexed like `kernels`: built on the first
    /// `ack_matmul_f16*` call at the widest subgroup the device can require,
    /// so a device that never runs an fp16 product never builds it and no
    /// error path in the open sequence moves.
    matmul_f16: Option<Vec<Kernel>>,
    /// The bf16-output set (2026-09-27), indexed a_t*2 + b_t: built on the
    /// first `ack_matmul_bf16_to_bf16` call at the widest subgroup the device
    /// can require.
    matmul_bf16_out: Option<Vec<Kernel>>,
    /// The fp16-output set (2026-09-30), indexed like the bf16-output one:
    /// built on the first `ack_matmul_f16_to_f16` call at the widest subgroup
    /// the device can require.
    matmul_f16_out: Option<Vec<Kernel>>,
    /// The bf16-storage loss modules (2026-09-28), [bf16 logits, bf16 logits and
    /// gradient]: built on the first `ack_loss_bf16` call.
    loss_bf16: Option<Vec<Kernel>>,
    /// rotary-f32/v1 (2026-09-29): built on the first `ack_rotary` call.
    rotary: Option<Kernel>,
    /// head-kl-ce-f16/v1 (2026-09-29): the forward and the backward, each built on
    /// its entry's first call.
    head_lse: Option<Kernel>,
    head_grad: Option<Kernel>,
    /// expert-factor-f16/v1 (2026-09-29): built on the first `ack_expert_factor_f16` call.
    expert_factor: Option<Kernel>,
    /// swiglu/v1 (2026-09-29), [FP32, bf16] storage: each built on the first
    /// `ack_swiglu` call that asks for it.
    swiglu: [Option<Kernel>; 2],
    /// weighted-bag/v1 (2026-10-22): built on the first `ack_weighted_bag` call.
    weighted_bag: Option<Kernel>,
    /// swiglu-scaled/v1 (2026-09-30), [FP32, bf16, fp16] storage: each built on
    /// the first `ack_swiglu_scaled` call that asks for it.
    swiglu_scaled: [Option<Kernel>; 3],
    /// student-norm-f16/v1 (2026-09-29): built on the first `ack_student_norm_f16` call.
    student_norm: Option<Kernel>,
    /// Families the public kit does not ship (see `DeviceExtension`), each
    /// added on its entry's first call.
    pub(crate) extensions: Vec<DeviceExtension>,
    casts: Vec<Kernel>, // f32->bf16, bf16->f32
    /// nf4-decode/v1 (2026-09-30), [FP32, bf16, fp16] output: each built on the
    /// first `ack_nf4_decode` call that asks for it.
    nf4_decode: [Option<Kernel>; 3],
    /// nf4-matmul/v1 (2026-09-30), [bf16, fp16, fp16 with the device's
    /// conversions] (`nf4_matmul_slot`), each [B_T = 0, 1]: built on the first
    /// `ack_matmul_nf4` call that needs it.
    nf4_matmul: [Option<Vec<Kernel>>; 3],
    /// nf4-matmul/v1 with a 16-bit C (2026-10-01), indexed as `nf4_matmul`:
    /// built on the first `ack_matmul_nf4_to_16` call that needs it.
    nf4_matmul_out16: [Option<Vec<Kernel>>; 3],
    /// nf4-gemv/v1 (2026-09-30), [bf16, fp16, fp16 with the device's
    /// conversions] x [1, 4, 16] rows a group: each built on the first
    /// `ack_gemv_nf4` call that needs it.
    nf4_gemv: [[Option<Kernel>; 3]; 3],
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
    pointwise_storage: [Option<Kernel>; pointwise_ops::STORAGE_MODULES - 1],
    /// The vector modules (see `SPV_POINTWISE_VECTOR`), each built with its
    /// storage module on the first call that asks for that storage.
    pointwise_vector: [Option<Kernel>; pointwise_ops::VECTOR_MODULES],
    /// `ACK_POINTWISE_VECTOR=0` was set when this device opened: every
    /// pointwise call takes its scalar module (a diagnostic, for A/B runs).
    pointwise_vector_off: bool,
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
    /// reduce-f32/v3 (2026-09-26), by storage module; see `SPV_REDUCE_V3`.
    reduce_v3: [Option<Kernel>; 4],
    /// The float64 module (schema 4); see `SPV_REDUCE_F64`.
    reduce_f64: Option<Kernel>,
    /// amp-unscale-f32/v1 (2026-09-29), built on the first `ack_amp_unscale`
    /// call; see `SPV_AMP_UNSCALE`.
    amp_unscale: Option<Kernel>,
    /// amp-update-scale-f32/v1 (2026-09-29), built on the first
    /// `ack_amp_update_scale` call; see `SPV_AMP_UPDATE`.
    amp_update: Option<Kernel>,
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
    /// index/v1 (ABI 12; four storage buffers, a 112-byte push block), built
    /// on the first `ack_index` call: a device that never gathers never builds
    /// it, and no error path in the open sequence moves.
    index: Option<Kernel>,
    /// convert/v1 (ABI 13; two storage buffers, a 72-byte push block), built
    /// on the first `ack_convert` call, as the index family's is.
    convert: Option<Kernel>,
    /// convert/v1's tiled transposed-read module (2026-10-03; the same
    /// bindings and push block), built on the first plan that takes it.
    convert_tiled: Option<Kernel>,
    /// intwise/v1 (ABI 14; three storage buffers, a 108-byte push block),
    /// built on the first `ack_intwise` call, as the convert family's is.
    intwise: Option<Kernel>,
    /// intwise/v1 schema 2's binary64 module (C = f64), built on the first
    /// `ack_intwise` call that asks for it.
    intwise_f64: Option<Kernel>,
    /// radix/v1 (intwise schema 6; three storage buffers, an 80-byte push
    /// block), built on the first `ack_intwise` call whose rank plans it.
    radix: Option<Kernel>,
    /// Optional vq4/v1 extension. Built only after a request passes live
    /// buffer admission, retired with the same context as every kernel.
    vq: Option<Kernel>,
    /// attention-v2: forward, dQ, dK/dV for head dimension 64 (slot 0) and
    /// 128 (slot 1), each set built on the first attention call at its head
    /// dimension, at the widest subgroup the device can require (64, else 32);
    /// a device that never runs attention never builds them. The first index is
    /// the element type (`ATTN_BF16`, `ATTN_F16`: attention-f16, 2026-09-29).
    attention: [[Option<[Kernel; 3]>; 2]; 2],
    buffers: HashMap<u32, Buffer>,
    next_buffer: u32,
    /// R4: staging areas created through `ack_staging_create`, keyed and
    /// handled exactly like `buffers` above (same packed-handle scheme,
    /// `staging_handle`/`split_staging_handle`).
    stagings: HashMap<u32, Staging>,
    next_staging: u32,
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
        // R4: staging areas are never recorded into a batch (see `Staging`'s
        // own doc comment), so destroying them before or after the discard
        // above makes no difference; done here, alongside the buffers, for
        // the same "clean up everything this device is holding" pass.
        for (_, s) in self.stagings.drain() {
            ctx.destroy_staging(s);
        }
        for (_, b) in self.buffers.drain() {
            ctx.destroy_buffer(b);
        }
        for (state, destroy) in self.extensions.drain(..) {
            destroy(state, ctx);
        }
        for k in self.kernels.drain(..) {
            ctx.destroy_kernel(k);
        }
        for k in self.matmul_f16.take().into_iter().flatten() {
            ctx.destroy_kernel(k);
        }
        for k in self.matmul_bf16_out.take().into_iter().flatten() {
            ctx.destroy_kernel(k);
        }
        for k in self.matmul_f16_out.take().into_iter().flatten() {
            ctx.destroy_kernel(k);
        }
        for k in self.loss_bf16.take().into_iter().flatten() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.rotary.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.head_lse.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.head_grad.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.expert_factor.take() {
            ctx.destroy_kernel(k);
        }
        for k in self.swiglu.iter_mut().filter_map(Option::take) {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.weighted_bag.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.student_norm.take() {
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
        for slot in self.pointwise_vector.iter_mut() {
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
            .chain(self.reduce_v3.iter_mut())
            .chain(std::iter::once(&mut self.reduce_f64))
            .chain(std::iter::once(&mut self.amp_unscale))
            .chain(std::iter::once(&mut self.amp_update))
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
        if let Some(k) = self.index.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.convert.take() {
            ctx.destroy_kernel(k);
        }
        if let Some(k) = self.convert_tiled.take() {
            ctx.destroy_kernel(k);
        }
        for slot in self.attention.iter_mut().flatten() {
            if let Some([fwd, dq, dkdv]) = slot.take() {
                ctx.destroy_kernel(fwd);
                ctx.destroy_kernel(dq);
                ctx.destroy_kernel(dkdv);
            }
        }
        if let Some(k) = self.vq.take() {
            ctx.destroy_kernel(k);
        }
        // (2026-09-30) these families were built and never destroyed: a closed
        // device left their pipelines, layouts and descriptor pools behind
        for k in [
            self.intwise.take(),
            self.intwise_f64.take(),
            self.radix.take(),
        ]
        .into_iter()
        .flatten()
        {
            ctx.destroy_kernel(k);
        }
        for k in self
            .swiglu_scaled
            .iter_mut()
            .chain(self.nf4_decode.iter_mut())
            .chain(self.nf4_gemv.iter_mut().flatten())
            .filter_map(Option::take)
        {
            ctx.destroy_kernel(k);
        }
        for k in self
            .nf4_matmul
            .iter_mut()
            .chain(self.nf4_matmul_out16.iter_mut())
            .filter_map(Option::take)
            .flatten()
        {
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
pub(crate) fn device(handle: *const AckDevice) -> Result<Arc<AckDevice>, i32> {
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
pub(crate) fn inner(dev: &AckDevice) -> Result<MutexGuard<'_, Inner>, i32> {
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
pub(crate) fn buffer_of<'a>(
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

/// A staging handle resolved against the device that must own it (R4). Same
/// three checks, same order, same codes as `buffer_of`.
fn staging_of<'a>(
    dev: &AckDevice,
    inner: &'a Inner,
    handle: *const Staging,
) -> Result<&'a Staging, i32> {
    let raw = handle as usize as u64;
    if raw == 0 {
        return Err(ACK_ERR_NULL);
    }
    let (owner, seq) = split_staging_handle(raw);
    if owner != dev.id {
        set_error(format!(
            "staging {raw:#x} belongs to device {owner}, not device {}",
            dev.id
        ));
        return Err(ACK_ERR_FOREIGN);
    }
    inner.stagings.get(&seq).ok_or_else(|| {
        set_error(format!(
            "staging {raw:#x} is not live on device {} (destroyed, or never allocated)",
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
            | ACK_OP_EMBEDDING
            | ACK_OP_MATMUL_F16
            | ACK_OP_INDEX
            | ACK_OP_CONVERT
            | ACK_OP_INTWISE,
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
        op_schema_matmul_f16: ACK_OP_SCHEMA_MATMUL_F16,
        op_schema_index: ACK_OP_SCHEMA_INDEX,
        op_schema_convert: ACK_OP_SCHEMA_CONVERT,
        op_schema_intwise: ACK_OP_SCHEMA_INTWISE,
    };
    // the caller promised `size` writable bytes, not alignment: write unaligned
    unsafe { std::ptr::write_unaligned(out, info) };
    ACK_OK
}

/// Open the device `ACK_DEVICE` chooses and build its kernels: the best GPU,
/// unless the caller asks for a CPU implementation of Vulkan by name
/// (`device_choice.rs`). Returns a handle, or null with `ack_last_error` set;
/// a host with no GPU driver gets a refusal saying so, not a CPU. The handle is an id: it must be closed with `ack_close`
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
    // casts only, and `ack_matmul_bf16` refuses on it by name (ABI 4).
    // Each set is compiled for ONE subgroup width, which its pipelines
    // require. 64 is tried first, so a device that admits it (RDNA) keeps the
    // modules it has always run; 32 is NVIDIA's only width. A device that can
    // guarantee neither opens without them, exactly as one without the
    // extension does.
    let rows_order = rows_order_forced();
    let coopmat_set: Option<(u32, Vec<&[u8]>)> =
        if !(ctx.subgroup_size_control && ctx.report.bf16_cooperative_matrix) {
            None
        } else if ctx.can_require_subgroup_size(64) && !narrow_lanes_forced() {
            let mut set = vec![
                SPV_NN, SPV_NT, SPV_TN, SPV_TT, SPV_E_NN, SPV_E_NT, SPV_E_TN, SPV_E_TT,
            ];
            if !rows_order {
                set.extend_from_slice(&SPV_RAST_S64);
            }
            Some((64, set))
        } else if ctx.can_require_subgroup_size(32) {
            Some((
                32,
                vec![
                    SPV_S32_NN,
                    SPV_S32_NT,
                    SPV_S32_TN,
                    SPV_S32_TT,
                    SPV_S32_E_NN,
                    SPV_S32_E_NT,
                    SPV_S32_E_TN,
                    SPV_S32_E_TT,
                ],
            ))
        } else {
            None
        };
    let mut kernels: Vec<Kernel> = Vec::with_capacity(RAST_AT + 4);
    if let Some((width, set)) = coopmat_set {
        // aligned NN, NT, TN, TT first, then the four edge variants at +4 and,
        // at 64 lanes, the four RAST variants at RAST_AT: `ack_matmul_bf16`
        // indexes this vector as edge*4 + a_t*2 + b_t, or by `rast_index`
        for spv in set {
            // the 32-byte push block: M, N, K and the five z words (ABI 10)
            match ctx.kernel_with_subgroup(spv, 3, COOPMAT_PUSH_BYTES as u32, Some(width)) {
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
            rows_order,
            matmul_f16: None,
            matmul_bf16_out: None,
            matmul_f16_out: None,
            loss_bf16: None,
            rotary: None,
            head_lse: None,
            head_grad: None,
            expert_factor: None,
            swiglu: [None, None],
            weighted_bag: None,
            extensions: Vec::new(),
            swiglu_scaled: [None, None, None],
            student_norm: None,
            casts,
            nf4_decode: [None, None, None],
            nf4_matmul: [None, None, None],
            nf4_matmul_out16: [None, None, None],
            nf4_gemv: [[None, None, None], [None, None, None], [None, None, None]],
            matmul_f32: Some(matmul_f32),
            matmul_f32_v2: Some(matmul_f32_v2),
            pointwise: Some(pointwise),
            pointwise_storage: [const { None }; pointwise_ops::STORAGE_MODULES - 1],
            pointwise_vector: [const { None }; pointwise_ops::VECTOR_MODULES],
            pointwise_vector_off: std::env::var_os("ACK_POINTWISE_VECTOR")
                .is_some_and(|v| v == "0"),
            reduce: Some(reduce),
            reduce_v2: Some(reduce_v2),
            reduce_storage: [const { None }; 3],
            reduce_v2_storage: [const { None }; 3],
            reduce_v3: [const { None }; 4],
            reduce_f64: None,
            amp_unscale: None,
            amp_update: None,
            reduce_v1_only: std::env::var_os("ACK_REDUCE_V1_ONLY").is_some_and(|v| v != "0"),
            rowwise: Some(rowwise),
            loss: Some(loss),
            embedding: Some(embedding),
            index: None,
            convert: None,
            convert_tiled: None,
            intwise: None,
            radix: None,
            intwise_f64: None,
            vq: None,
            attention: [[None, None], [None, None]],
            buffers: HashMap::new(),
            next_buffer: 1,
            stagings: HashMap::new(),
            next_staging: 1,
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
    if r.shader_float16 {
        features |= ACK_FEATURE_SHADER_FLOAT16;
    }
    if r.f16_cooperative_matrix {
        features |= ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP;
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

/// Charge the completion interval of every item recorded from now on to
/// `tag`, until the next call; 0 means untagged (`Context::set_dispatch_tag`).
/// With the timestamp instrument off it changes nothing that is measured.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move. NOT REFUSED WITH A BATCH OPEN: the tag is read per item as the
/// item is recorded, so one batch can hold items of several tags.
#[no_mangle]
pub extern "C" fn ack_set_dispatch_tag(dev: *const AckDevice, tag: u32) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    guard.ctx.set_dispatch_tag(tag);
    ACK_OK
}

/// Write up to `capacity` rows of the timestamp instrument's per-tag totals
/// (`Context::dispatch_tag_times`) to `out`, in tag order, and set `*count`
/// to the number of rows there are, so a caller whose `capacity` was short
/// learns how many and asks again. `out` may be null when `capacity` is 0 (a
/// count-only call). `size` is the caller's `AckDispatchTagTime` size and
/// must EQUAL this library's: the rows are written at this library's stride,
/// so a different layout is refused rather than handed misaligned rows.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move. READ-ONLY AND GENERATION-FREE, as `ack_dispatch_stats` is. It
/// takes no flush, so the rows describe the batches flushed so far; work
/// recorded and unsubmitted is in none of them.
/// # Safety
/// `out` must point to `capacity` writable rows of `size` bytes (or be null
/// with `capacity` 0), and `count` to one writable `usize`.
#[no_mangle]
pub unsafe extern "C" fn ack_dispatch_tag_times(
    dev: *const AckDevice,
    out: *mut AckDispatchTagTime,
    capacity: usize,
    size: usize,
    count: *mut usize,
) -> i32 {
    if count.is_null() || (out.is_null() && capacity > 0) {
        return ACK_ERR_NULL;
    }
    if size != std::mem::size_of::<AckDispatchTagTime>() {
        set_error(format!(
            "AckDispatchTagTime of {size} bytes is not this library's {}: the rows are written at the library's stride",
            std::mem::size_of::<AckDispatchTagTime>()
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
    let rows = guard.ctx.dispatch_tag_times();
    for (i, r) in rows.iter().take(capacity).enumerate() {
        let row = AckDispatchTagTime {
            size: size as u32,
            version: ACK_DISPATCH_TAG_TIME_VERSION,
            tag: r.tag,
            reserved: 0,
            items: r.items,
            device_ns: r.device_ns,
            max_item_ns: r.max_item_ns,
            unresolved_items: r.unresolved_items,
        };
        unsafe { std::ptr::write_unaligned(out.add(i), row) };
    }
    unsafe { std::ptr::write_unaligned(count, rows.len()) };
    ACK_OK
}

/// Fill `out` (of `size` bytes) with the context's flushes by reason and its
/// buffer and staging pools' counters (`AckRuntimeStats`). A struct smaller
/// than this library's is refused; a caller must require `size` to equal its
/// own struct's size and `version` to equal `ACK_RUNTIME_STATS_VERSION`.
///
/// READ-ONLY AND GENERATION-FREE, as `ack_dispatch_stats` is: it moves no
/// counter, changes no mode and takes no flush.
/// # Safety
/// `out` must point to `size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ack_runtime_stats(
    dev: *const AckDevice,
    out: *mut AckRuntimeStats,
    size: usize,
) -> i32 {
    if out.is_null() {
        return ACK_ERR_NULL;
    }
    if size < std::mem::size_of::<AckRuntimeStats>() {
        set_error(format!(
            "AckRuntimeStats of {size} bytes is smaller than this library's {}",
            std::mem::size_of::<AckRuntimeStats>()
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
    let reasons = ctx.flushes_by_reason();
    let mut flushes_by_reason = [0u64; ACK_FLUSH_REASON_SLOTS];
    flushes_by_reason[..reasons.len()].copy_from_slice(&reasons);
    let pool = ctx.pool_stats();
    let staging = ctx.staging_pool_stats();
    let stats = AckRuntimeStats {
        size: std::mem::size_of::<AckRuntimeStats>() as u32,
        version: ACK_RUNTIME_STATS_VERSION,
        flush_reason_count: reasons.len() as u32,
        batches_in_flight: ctx.batches_in_flight(),
        flushes_by_reason,
        pool_hits: pool.hits,
        pool_misses: pool.misses,
        pool_returns: pool.returns,
        pool_declined: pool.declined,
        pool_drained: pool.drained,
        pool_reclaims: pool.reclaims,
        pool_held_objects: pool.held_objects,
        pool_held_bytes: pool.held_bytes,
        staging_hits: staging.hits,
        staging_misses: staging.misses,
        staging_returns: staging.returns,
        staging_declined: staging.declined,
        staging_evicted: staging.evicted,
        staging_held_pairs: staging.held_pairs,
        staging_held_bytes: staging.held_bytes,
        max_recorded_items: ctx.max_recorded_items(),
        recorded_items: ctx.recorded_items(),
        max_retired_bytes: ctx.max_retired_bytes(),
        async_flush: u32::from(ctx.flush_is_asynchronous()),
        reserved: 0,
    };
    unsafe { std::ptr::write_unaligned(out, stats) };
    ACK_OK
}

/// The name of slot `index` of `AckRuntimeStats::flushes_by_reason`, as a
/// static NUL-terminated string, or null for a slot this library does not
/// use. It needs no device and takes no lock.
#[no_mangle]
pub extern "C" fn ack_flush_reason_name(index: u32) -> *const c_char {
    match FLUSH_REASON_C_NAMES.get(index as usize) {
        Some(name) => name.as_ptr(),
        None => std::ptr::null(),
    }
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

/// The shape block of `ack_attention_fwd` (128 bytes): batch, query heads,
/// kv heads, sequence length, head dimension (64), causal (0 or 1), scale, a
/// reserved word (0), then the (batch, head, row) strides in ELEMENTS of q, k,
/// v and o, the head dimension contiguous. See `attention_fused`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AckAttentionShape {
    pub batch: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub seq: u32,
    pub head_dim: u32,
    pub causal: u32,
    pub scale: f32,
    pub reserved: u32,
    pub q: [u64; 3],
    pub k: [u64; 3],
    pub v: [u64; 3],
    pub o: [u64; 3],
}

/// The block of `ack_attention_bwd` (224 bytes): the forward's shape, then the
/// strides of dO, dQ, dK and dV.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AckAttentionBackwardShape {
    pub forward: AckAttentionShape,
    pub dout: [u64; 3],
    pub dq: [u64; 3],
    pub dk: [u64; 3],
    pub dv: [u64; 3],
}

fn strides(s: [u64; 3]) -> attention_fused::Strides {
    attention_fused::Strides {
        batch: s[0],
        head: s[1],
        row: s[2],
    }
}

fn attention_shape(a: &AckAttentionShape) -> Result<attention_fused::AttentionShape, i32> {
    if a.reserved != 0 || a.causal > 1 {
        set_error("attention: the shape block's reserved word must be 0 and causal 0 or 1");
        return Err(ACK_ERR_SHAPE);
    }
    Ok(attention_fused::AttentionShape {
        batch: a.batch,
        heads: a.heads,
        kv_heads: a.kv_heads,
        seq: a.seq,
        head_dim: a.head_dim,
        causal: a.causal == 1,
        scale: a.scale,
        q: strides(a.q),
        k: strides(a.k),
        v: strides(a.v),
        o: strides(a.o),
    })
}

/// The kernel set of a head dimension (slot 0: 64, slot 1: 128), refused by
/// name for any other.
fn attention_slot(head_dim: u32) -> Result<usize, i32> {
    match head_dim {
        64 => Ok(0),
        128 => Ok(1),
        d => {
            set_error(format!(
                "attention: {}",
                attention_fused::AttentionPlanError::HeadDim(d)
            ));
            Err(ACK_ERR_SHAPE)
        }
    }
}

/// Build one (element type, head dimension)'s three pipelines at the widest
/// subgroup the device can require, on the first attention call there.
fn attention_kernels(state: &mut Inner, elt: usize, dim: usize) -> Result<(), i32> {
    if state.attention[elt][dim].is_some() {
        return Ok(());
    }
    let (slot, width) = if state.ctx.can_require_subgroup_size(64) && !narrow_lanes_forced() {
        (0, 64)
    } else if state.ctx.can_require_subgroup_size(32) {
        (1, 32)
    } else {
        set_error("attention: the device can require neither 64 nor 32 lanes");
        return Err(ACK_ERR_UNSUPPORTED_OP);
    };
    let hardware = elt == ATTN_F16 && attention_f16_hardware(&state.ctx);
    let table: &[[[&[u8]; 2]; 2]; 3] = if hardware {
        &ATTN_SPV_F16HW
    } else {
        &ATTN_SPV[elt]
    };
    let build = |spv: &[u8], operands: usize, push: usize| {
        state
            .ctx
            .kernel_with_subgroup(spv, operands as u32, push as u32, Some(width))
    };
    let built = (|| {
        let fwd = build(
            table[0][dim][slot],
            attention_fused::OPERANDS,
            attention_fused::PUSH_BYTES,
        )?;
        let dq = match build(
            table[1][dim][slot],
            attention_fused::DQ_OPERANDS,
            attention_fused::DQ_PUSH_BYTES,
        ) {
            Ok(k) => k,
            Err(e) => {
                state.ctx.destroy_kernel(fwd);
                return Err(e);
            }
        };
        match build(
            table[2][dim][slot],
            attention_fused::DKDV_OPERANDS,
            attention_fused::DKDV_PUSH_BYTES,
        ) {
            Ok(dkdv) => Ok([fwd, dq, dkdv]),
            Err(e) => {
                state.ctx.destroy_kernel(fwd);
                state.ctx.destroy_kernel(dq);
                Err(e)
            }
        }
    })();
    match built {
        Ok(k) => {
            state.attention[elt][dim] = Some(k);
            Ok(())
        }
        Err(e) => {
            set_error(format!(
                "attention pipelines ({}{}, {width} lanes): {e}",
                ATTN_ELT_NAMES[elt],
                if hardware {
                    ", hardware conversion"
                } else {
                    ""
                }
            ));
            Err(code_for(&e))
        }
    }
}

/// A gradient or output buffer that is also an input would be read and written
/// by one dispatch; refused by name before anything is recorded.
fn attention_disjoint(outputs: &[&Buffer], inputs: &[&Buffer]) -> Result<(), i32> {
    for o in outputs {
        for i in inputs {
            if o.id() == i.id() {
                set_error("attention: attention-output-aliases-input");
                return Err(ACK_ERR_SHAPE);
            }
        }
    }
    for (n, a) in outputs.iter().enumerate() {
        for b in &outputs[n + 1..] {
            if a.id() == b.id() {
                set_error("attention: attention-outputs-alias");
                return Err(ACK_ERR_SHAPE);
            }
        }
    }
    Ok(())
}

/// The forward of either element type (`ack_attention_fwd`, `ack_attention_f16_fwd`).
#[allow(clippy::too_many_arguments)]
unsafe fn attention_fwd_elt(
    elt: usize,
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    lse: *const Buffer,
    shape: *const AckAttentionShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    if shape.is_null() {
        return ACK_ERR_NULL;
    }
    let shape = match attention_shape(unsafe { &*shape }) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let dim = match attention_slot(shape.head_dim) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = attention_kernels(&mut guard, elt, dim) {
        return code;
    }
    let mut resolved = Vec::with_capacity(5);
    for h in [q, k, v, o, lse] {
        match buffer_of(&device, &guard, h) {
            Ok(b) => resolved.push(b),
            Err(code) => return code,
        }
    }
    let [q, k, v, o, lse] = [
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
    ];
    if let Err(code) = attention_disjoint(&[o, lse], &[q, k, v]) {
        return code;
    }
    let plan = match attention_fused::AttentionPlan::new(
        shape,
        [
            q.bytes / 2,
            k.bytes / 2,
            v.bytes / 2,
            o.bytes / 2,
            lse.bytes / 4,
        ],
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("attention: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let kernel = &guard.attention[elt][dim].as_ref().expect("built above")[0];
    if let Err(e) = guard.ctx.bind(kernel, &[q, k, v, o, lse]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(q.id()),
        Access::read(k.id()),
        Access::read(v.id()),
        Access::write(o.id()),
        Access::write(lse.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// The backward of either element type (`ack_attention_bwd`, `ack_attention_f16_bwd`).
#[allow(clippy::too_many_arguments)]
unsafe fn attention_bwd_elt(
    elt: usize,
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    dout: *const Buffer,
    lse: *const Buffer,
    dq: *const Buffer,
    dk: *const Buffer,
    dv: *const Buffer,
    block: *const AckAttentionBackwardShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    if block.is_null() {
        return ACK_ERR_NULL;
    }
    let block = unsafe { &*block };
    let shape = match attention_shape(&block.forward) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let dim = match attention_slot(shape.head_dim) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = attention_kernels(&mut guard, elt, dim) {
        return code;
    }
    let mut resolved = Vec::with_capacity(9);
    for h in [q, k, v, o, dout, lse, dq, dk, dv] {
        match buffer_of(&device, &guard, h) {
            Ok(b) => resolved.push(b),
            Err(code) => return code,
        }
    }
    let [q, k, v, o, g, lse, dq, dk, dv] = [
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
        resolved[5],
        resolved[6],
        resolved[7],
        resolved[8],
    ];
    if let Err(code) = attention_disjoint(&[dq, dk, dv], &[q, k, v, o, g, lse]) {
        return code;
    }
    let forward = match attention_fused::AttentionPlan::new(
        shape,
        [
            q.bytes / 2,
            k.bytes / 2,
            v.bytes / 2,
            o.bytes / 2,
            lse.bytes / 4,
        ],
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("attention: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let bw = match attention_fused::AttentionBackward::new(
        forward,
        attention_fused::BackwardStrides {
            dout: strides(block.dout),
            dq: strides(block.dq),
            dk: strides(block.dk),
            dv: strides(block.dv),
        },
        [g.bytes / 2, dq.bytes / 2, dk.bytes / 2, dv.bytes / 2],
    ) {
        Ok(b) => b,
        Err(e) => {
            set_error(format!("attention backward: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let kernels = guard.attention[elt][dim].as_ref().expect("built above");
    let (dqk, dkdvk) = (&kernels[1], &kernels[2]);
    let r = Access::read;
    let w = Access::write;
    if let Err(e) = guard.ctx.bind(dqk, &[q, k, v, o, g, lse, dq]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let rc = recorded(guard.ctx.dispatch(
        dqk,
        &bw.dq_push_constants(),
        bw.dq_groups(),
        1,
        &[
            r(q.id()),
            r(k.id()),
            r(v.id()),
            r(o.id()),
            r(g.id()),
            r(lse.id()),
            w(dq.id()),
        ],
    ));
    if rc != ACK_OK {
        return rc;
    }
    if let Err(e) = guard.ctx.bind(dkdvk, &[q, k, v, o, g, lse, dk, dv]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    recorded(guard.ctx.dispatch(
        dkdvk,
        &bw.dkdv_push_constants(),
        bw.dkdv_groups(),
        1,
        &[
            r(q.id()),
            r(k.id()),
            r(v.id()),
            r(o.id()),
            r(g.id()),
            r(lse.id()),
            w(dk.id()),
            w(dv.id()),
        ],
    ))
}

/// Fused causal attention, forward (attention-v2): o = softmax(scale q k^T +
/// mask) v and lse, per (batch, query head), bf16 q, k, v and o of head
/// dimension 64 or 128, f32 lse of batch * heads * seq elements. RECORDED like every operator entry:
/// `ms_out` is NaN. Refused by name through `attention_fused::AttentionPlan`.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move, and a caller treats its absence as "no fused attention".
/// # Safety
/// Every handle must come from this device; `shape` must point to an
/// `AckAttentionShape`; `ms_out` must be null or point to a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_attention_fwd(
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    lse: *const Buffer,
    shape: *const AckAttentionShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe { attention_fwd_elt(ATTN_BF16, dev, q, k, v, o, lse, shape, ms_out) }
}

/// Fused causal attention, backward (attention-v2): dq, dk and dv from
/// the forward's q, k, v, o and lse and the upstream do, in two recorded
/// dispatches (dQ over query blocks, dK/dV over key blocks; no atomics).
/// OPTIONAL like `ack_attention_fwd`. `ms_out` is NaN.
/// # Safety
/// As `ack_attention_fwd`; `block` must point to an
/// `AckAttentionBackwardShape`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_attention_bwd(
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    dout: *const Buffer,
    lse: *const Buffer,
    dq: *const Buffer,
    dk: *const Buffer,
    dv: *const Buffer,
    block: *const AckAttentionBackwardShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe {
        attention_bwd_elt(
            ATTN_BF16, dev, q, k, v, o, dout, lse, dq, dk, dv, block, ms_out,
        )
    }
}

/// Fused causal attention over BINARY16 (attention-f16, 2026-09-29), forward:
/// `ack_attention_fwd`'s contract, shape block, refusals and recording, with
/// fp16 q, k, v and o (f32 lse), the fp16 autocast stage E is registered with.
/// The kernels are attention-v2's built with -DELT_F16=1: S and O accumulate in
/// f32 and the block's probabilities are rounded to binary16 for the P V
/// product, by exact integer routines. OPTIONAL, resolved by name like
/// `ack_attention_fwd`: a caller treats its absence as "no fp16 fused attention".
/// # Safety
/// As `ack_attention_fwd`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_attention_f16_fwd(
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    lse: *const Buffer,
    shape: *const AckAttentionShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe { attention_fwd_elt(ATTN_F16, dev, q, k, v, o, lse, shape, ms_out) }
}

/// Fused causal attention over BINARY16 (attention-f16, 2026-09-29), backward:
/// `ack_attention_bwd`'s contract over fp16 q, k, v, o, do, dq, dk and dv (f32
/// lse), dS rounded to binary16 for its products. OPTIONAL like
/// `ack_attention_f16_fwd`.
/// # Safety
/// As `ack_attention_bwd`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_attention_f16_bwd(
    dev: *const AckDevice,
    q: *const Buffer,
    k: *const Buffer,
    v: *const Buffer,
    o: *const Buffer,
    dout: *const Buffer,
    lse: *const Buffer,
    dq: *const Buffer,
    dk: *const Buffer,
    dv: *const Buffer,
    block: *const AckAttentionBackwardShape,
    ms_out: *mut f64,
) -> i32 {
    unsafe {
        attention_bwd_elt(
            ATTN_F16, dev, q, k, v, o, dout, lse, dq, dk, dv, block, ms_out,
        )
    }
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
///
/// The copy lands straight in `data`, a chunk of at most
/// `READBACK_CHUNK_BYTES` at a time (`Context::download_into`). A size
/// refusal writes nothing; after any other failure the first bytes of `data`
/// may already hold the buffer's and the rest are unchanged, so treat the
/// whole of `data` as unspecified.
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
    // straight into the caller's bytes: no intermediate Vec (download_into)
    let out = unsafe { std::slice::from_raw_parts_mut(data, len) };
    match guard.ctx.download_into(buffer, out) {
        Ok(()) => ACK_OK,
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

// ---- R4: the staging path (dedicated transfer queue, timeline semaphores,
// bindable staging), so the torch backend can reach the 6.4x-faster measured
// path -- see the crate's Rust `Context::staging`/`copy_from_staging_async`
// for what each of these wraps. -------------------------------------------

/// `kind` 0 is `HostCoherent`, 1 is `HostCached`, 2 is `DeviceLocalHostVisible`.
fn staging_kind_of(kind: u32) -> Result<StagingKind, i32> {
    match kind {
        0 => Ok(StagingKind::HostCoherent),
        1 => Ok(StagingKind::HostCached),
        2 => Ok(StagingKind::DeviceLocalHostVisible),
        _ => {
            set_error(format!("staging kind {kind} is not 0, 1 or 2"));
            Err(ACK_ERR_SHAPE)
        }
    }
}

/// A persistent, mapped staging area of `bytes`, of the given `kind` (see
/// `staging_kind_of`); null on failure with `ack_last_error` set. Mirrors
/// `ack_buffer_alloc`'s handle convention exactly: the returned pointer names
/// this device and this staging area and is refused, not dereferenced, on any
/// other device or after `ack_staging_destroy`.
#[no_mangle]
pub extern "C" fn ack_staging_create(dev: *const AckDevice, bytes: u64, kind: u32) -> *mut Staging {
    if bytes == 0 {
        set_error("zero-byte staging area");
        return std::ptr::null_mut();
    }
    let kind = match staging_kind_of(kind) {
        Ok(k) => k,
        Err(_) => return std::ptr::null_mut(),
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(_) => return std::ptr::null_mut(),
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(_) => return std::ptr::null_mut(),
    };
    if guard.next_staging == u32::MAX {
        set_error("staging handle space exhausted on this device; close and reopen");
        return std::ptr::null_mut();
    }
    match guard.ctx.staging(bytes, kind) {
        Ok(s) => {
            let seq = guard.next_staging;
            guard.next_staging += 1;
            guard.stagings.insert(seq, s);
            staging_handle(device.id, seq) as usize as *mut Staging
        }
        Err(e) => {
            set_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Destroy a staging area. `ACK_ERR_FOREIGN` for another device's, `ACK_ERR_FREED`
/// for one already destroyed; mirrors `ack_buffer_free`'s checks exactly. Unlike
/// a buffer this is never retired into a batch (`Staging` is never recorded),
/// so `ACK_OK` here does mean the Vulkan objects are gone.
#[no_mangle]
pub extern "C" fn ack_staging_destroy(dev: *const AckDevice, staging: *mut Staging) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = staging_of(&device, &guard, staging) {
        return code;
    }
    let (_, seq) = split_staging_handle(staging as usize as u64);
    if let Some(s) = guard.stagings.remove(&seq) {
        guard.ctx.destroy_staging(s);
    }
    ACK_OK
}

/// The staging area's host pointer and length, so the torch backend can land
/// bytes into it directly (an unbuffered file read, or a plain memcpy) without
/// a fresh allocate-map-copy-unmap per call. `*out_ptr` is writable: every
/// `StagingKind` `Context::staging` produces is `HOST_COHERENT`, so a write
/// through it needs no flush before `ack_copy_from_staging`/
/// `ack_copy_from_staging_async` reads it.
/// # Safety
/// `out_ptr` and `out_len` must be null or point to a writable `*mut u8` /
/// `u64` respectively.
#[no_mangle]
pub unsafe extern "C" fn ack_staging_map(
    dev: *const AckDevice,
    staging: *const Staging,
    out_ptr: *mut *mut u8,
    out_len: *mut u64,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
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
    let s = match staging_of(&device, &guard, staging) {
        Ok(s) => s,
        Err(code) => return code,
    };
    unsafe {
        *out_ptr = s.as_ptr() as *mut u8;
        *out_len = s.bytes();
    }
    ACK_OK
}

/// Copy `bytes` from the staging area (at `src_offset`) into a device buffer
/// (at `dst_offset`). Synchronous: returns once the copy has completed. Wraps
/// `Context::copy_from_staging`; refusals mirror it exactly.
#[no_mangle]
pub extern "C" fn ack_copy_from_staging(
    dev: *const AckDevice,
    staging: *const Staging,
    src_offset: u64,
    dst: *const Buffer,
    dst_offset: u64,
    bytes: u64,
) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let s = match staging_of(&device, &guard, staging) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let dst = match buffer_of(&device, &guard, dst) {
        Ok(b) => b,
        Err(code) => return code,
    };
    match guard
        .ctx
        .copy_from_staging(s, src_offset, dst, dst_offset, bytes)
    {
        Ok(()) => ACK_OK,
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

/// Record the copy on the transfer queue and return at once, without waiting
/// (R2's fallback applies transparently on a device with no transfer queue or
/// no timeline semaphore -- this still works there, synchronously). `*ticket_out`
/// receives the ticket's raw value: an opaque `u64` at this boundary, passed
/// back to `ack_await_transfer`/`ack_wait_transfer` unchanged. Wraps
/// `Context::copy_from_staging_async`; refusals mirror it exactly.
/// # Safety
/// `ticket_out` must be null or point to a writable `u64`.
#[no_mangle]
pub unsafe extern "C" fn ack_copy_from_staging_async(
    dev: *const AckDevice,
    staging: *const Staging,
    src_offset: u64,
    dst: *const Buffer,
    dst_offset: u64,
    bytes: u64,
    ticket_out: *mut u64,
) -> i32 {
    if ticket_out.is_null() {
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
    let s = match staging_of(&device, &guard, staging) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let dst = match buffer_of(&device, &guard, dst) {
        Ok(b) => b,
        Err(code) => return code,
    };
    match guard
        .ctx
        .copy_from_staging_async(s, src_offset, dst, dst_offset, bytes)
    {
        Ok(ticket) => {
            unsafe { *ticket_out = ticket.value() };
            ACK_OK
        }
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

/// Make every dispatch this device records after this call wait on `ticket`
/// before it reads whatever the copy behind it wrote -- NOT a host-side block
/// (see `Context::await_transfer`). A no-op, returning `ACK_OK`, on a device
/// with no timeline semaphore.
#[no_mangle]
pub extern "C" fn ack_await_transfer(dev: *const AckDevice, ticket: u64) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    match guard.ctx.await_transfer(&TransferTicket::from_raw(ticket)) {
        Ok(()) => ACK_OK,
        Err(e) => {
            set_error(format!("{e}"));
            code_for(&e)
        }
    }
}

/// Host-side block until `ticket`'s copy has completed (see
/// `Context::wait_transfer`): for the torch backend's teardown, or anywhere a
/// real wait rather than an ordering against the next dispatch is wanted.
/// Returns at once if already signalled, and at once, unconditionally, on a
/// device with no timeline semaphore.
#[no_mangle]
pub extern "C" fn ack_wait_transfer(dev: *const AckDevice, ticket: u64) -> i32 {
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    match guard.ctx.wait_transfer(&TransferTicket::from_raw(ticket)) {
        Ok(()) => ACK_OK,
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
    record_cast(&guard, src, dst, n, mode)
}

/// Record `ack_cast`'s dispatch on resolved buffers (shared with the families
/// that record the same operations: src/moe_backward.rs): `mode` 0 is f32 ->
/// bf16 (round to nearest even), 1 is bf16 -> f32, over `n` elements. Refuses
/// by name what `ack_cast` refuses about the count, the mode and the sizes.
/// Returns `ACK_OK` once RECORDED.
pub(crate) fn record_cast(state: &Inner, src: &Buffer, dst: &Buffer, n: u64, mode: i32) -> i32 {
    if n == 0 || n > u32::MAX as u64 || !(mode == 0 || mode == 1) {
        set_error(format!("cast: bad n={n} or mode={mode}"));
        return ACK_ERR_SHAPE;
    }
    let (src_bytes, dst_bytes) = if mode == 0 {
        (n * 4, n * 2)
    } else {
        (n * 2, n * 4)
    };
    if src.bytes < src_bytes || dst.bytes < dst_bytes {
        set_error(format!("cast: buffers too small for n={n} mode={mode}"));
        return ACK_ERR_SIZE;
    }
    let kernel = &state.casts[mode as usize];
    if let Err(e) = state.ctx.bind(kernel, &[src, dst]) {
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
    recorded(state.ctx.dispatch(kernel, &push, groups, 1, &access))
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
    unsafe { matmul16(Operand16::Bf16, dev, a, b, c, m, n, k, a_t, b_t, ms_out) }
}

/// `ack_matmul_bf16` with fp16 operands (ABI 11, operator bit 256, schema 1):
/// the same signature, shapes, layouts, buffer sizes (two bytes an operand
/// element, four an output element) and refusals, running the fp16 modules
/// (`-DELEM_F16=1`) with the f32 accumulator and output unchanged. For a
/// device whose cooperative matrices take fp16 and not bf16, such as NVIDIA
/// Turing. The set is built on the first fp16 call; a device without the
/// 16x16x16 fp16 -> f32 subgroup shape, `shaderFloat16` or subgroup size
/// control refuses with `ACK_ERR_UNSUPPORTED_OP` before its buffers are looked
/// at. The dispatch is RECORDED and runs at the next flush, so `ms_out` is NaN.
/// # Safety
/// `ms_out` must be null or point to a writable f64.
#[no_mangle]
pub unsafe extern "C" fn ack_matmul_f16(
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
    unsafe { matmul16(Operand16::F16, dev, a, b, c, m, n, k, a_t, b_t, ms_out) }
}

/// The body of `ack_matmul_bf16` and `ack_matmul_f16`: one product through the
/// 16-bit operand family's modules.
#[allow(clippy::too_many_arguments)]
unsafe fn matmul16(
    operand: Operand16,
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
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // the fp16 set is built on its first product, and before the buffers are
    // resolved: they borrow the guard. A device without fp16 cooperative
    // matrices is refused here, by code, before anything else is looked at
    if operand == Operand16::F16 {
        if let Err(code) = matmul_f16_kernels(&mut guard) {
            return code;
        }
    }
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
    let kernels = match operand {
        Operand16::Bf16 => {
            if guard.kernels.is_empty() {
                set_error(
                    "BF16 cooperative-matrix matmul is not available on this device (no BF16 cooperative matrices or no subgroup size control); ack_matmul_f32 is",
                );
                return ACK_ERR_UNSUPPORTED_OP;
            }
            &guard.kernels
        }
        Operand16::F16 => guard.matmul_f16.as_ref().expect("built above"),
    };
    let index = rast_index(!edge, n, kernels.len(), a_t, b_t)
        .unwrap_or((edge as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize));
    let kernel = &kernels[index];
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
    unsafe {
        matmul16_z(
            Operand16::Bf16,
            dev,
            a,
            b,
            c,
            m,
            n,
            k,
            a_t,
            b_t,
            z,
            zmode,
            za,
            zb,
            zc,
            kz,
            lda,
            ldb,
            ms_out,
        )
    }
}

/// `ack_matmul_bf16_z` with fp16 operands (ABI 11, operator bit 256, schema 1):
/// the same z grid, strides, pitches and refusals as the bf16 entry, through
/// the fp16 modules, with the same first-call build and refusal as
/// `ack_matmul_f16`. A call with `z = 1` and every z word zero is
/// `ack_matmul_f16`. The dispatch is RECORDED, so `ms_out` is NaN.
/// # Safety
/// `ms_out` must be null or point to a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_f16_z(
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
    unsafe {
        matmul16_z(
            Operand16::F16,
            dev,
            a,
            b,
            c,
            m,
            n,
            k,
            a_t,
            b_t,
            z,
            zmode,
            za,
            zb,
            zc,
            kz,
            lda,
            ldb,
            ms_out,
        )
    }
}

/// The body of `ack_matmul_bf16_z` and `ack_matmul_f16_z`.
#[allow(clippy::too_many_arguments)]
unsafe fn matmul16_z(
    operand: Operand16,
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
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // the fp16 set is built on its first product, and before the buffers are
    // resolved: they borrow the guard. A device without fp16 cooperative
    // matrices is refused here, by code, before anything else is looked at
    if operand == Operand16::F16 {
        if let Err(code) = matmul_f16_kernels(&mut guard) {
            return code;
        }
    }
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
    let kernels = match operand {
        Operand16::Bf16 => {
            if guard.kernels.is_empty() {
                set_error(
                    "BF16 cooperative-matrix matmul is not available on this device (no BF16 cooperative matrices or no subgroup size control); ack_matmul_f32 is",
                );
                return ACK_ERR_UNSUPPORTED_OP;
            }
            &guard.kernels
        }
        Operand16::F16 => guard.matmul_f16.as_ref().expect("built above"),
    };
    let index = rast_index(!edge, n, kernels.len(), a_t, b_t)
        .unwrap_or((edge as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize));
    let kernel = &kernels[index];
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

/// Which 16-bit operand family a cooperative-matrix product runs (ABI 11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operand16 {
    Bf16,
    F16,
}

/// Build the fp16 set on the first fp16 product, at the widest subgroup the
/// device can require: 64 first, so a device that admits it (RDNA) runs the
/// wave64 modules the bf16 set would, then 32, NVIDIA's only width. Refused by
/// code, and nothing built, on a device without the 16x16x16 fp16 -> f32
/// subgroup shape, `shaderFloat16` (the modules declare `Float16`) or
/// subgroup size control. A failed build destroys what it built and is
/// retried on the next call.
fn matmul_f16_kernels(state: &mut Inner) -> Result<(), i32> {
    if state.matmul_f16.is_some() {
        return Ok(());
    }
    let report = &state.ctx.report;
    if !(state.ctx.subgroup_size_control && report.shader_float16 && report.f16_cooperative_matrix)
    {
        set_error(
            "FP16 cooperative-matrix matmul is not available on this device (no 16x16x16 fp16 -> f32 subgroup cooperative matrices, no shaderFloat16, or no subgroup size control); ack_matmul_f32 is",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    let (set, width): (&[&[u8]], u32) =
        if state.ctx.can_require_subgroup_size(64) && !narrow_lanes_forced() {
            let with_rast = if state.rows_order {
                RAST_AT
            } else {
                SPV_F16_S64.len()
            };
            (&SPV_F16_S64[..with_rast], 64)
        } else if state.ctx.can_require_subgroup_size(32) {
            (&SPV_F16_S32, 32)
        } else {
            set_error("fp16 matmul: the device can require neither 64 nor 32 lanes");
            return Err(ACK_ERR_UNSUPPORTED_OP);
        };
    let mut built: Vec<Kernel> = Vec::with_capacity(set.len());
    for spv in set.iter() {
        // the same 32-byte push block as the bf16 set: M, N, K and the z words
        match state
            .ctx
            .kernel_with_subgroup(spv, 3, COOPMAT_PUSH_BYTES as u32, Some(width))
        {
            Ok(k) => built.push(k),
            Err(e) => {
                for k in built.drain(..) {
                    state.ctx.destroy_kernel(k);
                }
                set_error(format!("fp16 matmul pipeline ({width} lanes): {e}"));
                return Err(code_for(&e));
            }
        }
    }
    state.matmul_f16 = Some(built);
    Ok(())
}

/// Build the bf16-output product set if this device has not built it yet
/// (`ack_matmul_bf16_to_bf16`'s first half, shared with src/moe_backward.rs).
/// On a refusal the message is set and the code returned.
pub(crate) fn matmul_bf16_out_kernels(state: &mut Inner) -> Result<(), i32> {
    if state.matmul_bf16_out.is_some() {
        return Ok(());
    }
    if state.kernels.is_empty() {
        set_error(
            "BF16 cooperative-matrix matmul is not available on this device (no BF16 cooperative matrices or no subgroup size control); ack_matmul_f32 is",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    // the four edge modules, then the four aligned ones: index aligned*4 + a_t*2 + b_t;
    // at 64 lanes the four RAST ones follow at RAST_AT (`rast_index`)
    let (edge, aligned, rast, width): (_, _, &[&[u8]], _) =
        if state.ctx.can_require_subgroup_size(64) && !narrow_lanes_forced() {
            (
                &SPV_BF16_OUT_S64,
                &SPV_BF16_OUT_ALIGNED_S64,
                if state.rows_order {
                    &[]
                } else {
                    &SPV_BF16_OUT_RAST_S64
                },
                64,
            )
        } else if state.ctx.can_require_subgroup_size(32) {
            (&SPV_BF16_OUT_S32, &SPV_BF16_OUT_ALIGNED_S32, &[], 32)
        } else {
            set_error("bf16-output matmul: the device can require neither 64 nor 32 lanes");
            return Err(ACK_ERR_UNSUPPORTED_OP);
        };
    let mut built: Vec<Kernel> = Vec::with_capacity(RAST_AT + 4);
    for spv in edge.iter().chain(aligned.iter()).chain(rast.iter()) {
        match state
            .ctx
            .kernel_with_subgroup(spv, 3, COOPMAT_PUSH_BYTES as u32, Some(width))
        {
            Ok(k) => built.push(k),
            Err(e) => {
                for k in built.drain(..) {
                    state.ctx.destroy_kernel(k);
                }
                set_error(format!("bf16-output matmul pipeline ({width} lanes): {e}"));
                return Err(code_for(&e));
            }
        }
    }
    state.matmul_bf16_out = Some(built);
    Ok(())
}

/// `C = A . B` with bf16 operands and a BF16 `C` (2026-09-27): the modules
/// compiled with -DBF16_OUT=1 (the aligned ones for a shape on the 128 x 128 x 32
/// tile since 2026-09-28, the edge ones otherwise), so the f32 accumulator is
/// rounded to nearest even in the product's own epilogue, bit for bit what
/// `ack_matmul_bf16` followed by `ack_cast` (mode 0) writes -- without the f32
/// block or the cast dispatch. `n` must be even: C is written in bf16 pairs,
/// and a pair must not straddle two rows. RECORDED like every operator entry.
///
/// OPTIONAL, resolved by name and in no pointer table, like the attention
/// entries: `ACK_ABI_VERSION` does not move, and a caller treats its absence
/// as "cast after the product".
/// # Safety
/// Every handle must come from this device; `ms_out` must be null or point to
/// a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_bf16_to_bf16(
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
    unsafe { no_duration(ms_out) };
    if m == 0 || n == 0 || k == 0 {
        set_error(format!(
            "shape {m}x{n}x{k} has a zero dimension: the dispatch grid would be empty"
        ));
        return ACK_ERR_SHAPE;
    }
    if n % 2 != 0 {
        set_error(format!(
            "matmul-bf16-out-odd-n: n = {n}; C is written in bf16 pairs, which an odd row would straddle"
        ));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = matmul_bf16_out_kernels(&mut guard) {
        return code;
    }
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
    record_matmul_bf16_out(&guard, a, b, c, m, n, k, a_t, b_t)
}

/// Record `ack_matmul_bf16_to_bf16`'s dispatch on resolved buffers (its second
/// half, shared with src/moe_backward.rs): the sizes and aliasing it refuses,
/// then the module it chooses for the shape and layout. `m`, `n` and `k` must
/// be nonzero and `n` even (the entry refuses those before any lookup), and
/// `matmul_bf16_out_kernels` must have built the set. Returns `ACK_OK` once
/// RECORDED.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_matmul_bf16_out(
    state: &Inner,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: u32,
    n: u32,
    k: u32,
    a_t: i32,
    b_t: i32,
) -> i32 {
    let need_a = m as u64 * k as u64 * 2;
    let need_b = k as u64 * n as u64 * 2;
    let need_c = m as u64 * n as u64 * 2;
    if a.bytes < need_a || b.bytes < need_b || c.bytes < need_c {
        set_error(format!(
            "buffers too small for {m}x{n}x{k}: A {} (need {need_a}), B {} (need {need_b}), C {} (need {need_c})",
            a.bytes, b.bytes, c.bytes
        ));
        return ACK_ERR_SIZE;
    }
    if c.id() == a.id() || c.id() == b.id() {
        set_error("matmul-bf16-out-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let Some(set) = state.matmul_bf16_out.as_ref() else {
        set_error("matmul-bf16-out: the product set is not built on this device");
        return ACK_ERR_UNSUPPORTED_OP;
    };
    let aligned = bf16_out_aligned(m, n, k);
    let index = rast_index(aligned, n, set.len(), a_t, b_t)
        .unwrap_or((aligned as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize));
    let kernel = &set[index];
    if let Err(e) = state.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let push = coopmat_push(m, n, k, 0, 0, 0, 0, 0, 0, 0);
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    recorded(state.ctx.dispatch(
        kernel,
        &push,
        [n.div_ceil(TILE), m.div_ceil(TILE), 1],
        1,
        &access,
    ))
}

fn matmul_f16_out_kernels(state: &mut Inner) -> Result<(), i32> {
    if state.matmul_f16_out.is_some() {
        return Ok(());
    }
    let report = &state.ctx.report;
    if !(state.ctx.subgroup_size_control && report.shader_float16 && report.f16_cooperative_matrix)
    {
        set_error(
            "FP16 cooperative-matrix matmul is not available on this device (no 16x16x16 fp16 -> f32 subgroup cooperative matrices, no shaderFloat16, or no subgroup size control); ack_matmul_f32 is",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    // the modules round with the device's own conversion under declared float
    // controls; a device that cannot honour them does not build them
    if !state.ctx.fp16_rounding_declarable {
        set_error(
            "matmul-f16-out-float-controls: the fp16-output product declares RoundingModeRTE and DenormPreserve for 16- and 64-bit floats, and this device does not report all four; ack_matmul_f16 and a conversion are",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    // the four edge modules, then the four aligned ones: index aligned*4 + a_t*2 + b_t;
    // at 64 lanes the four RAST ones follow at RAST_AT (`rast_index`)
    let (edge, aligned, rast, width): (_, _, &[&[u8]], _) =
        if state.ctx.can_require_subgroup_size(64) && !narrow_lanes_forced() {
            (
                &SPV_F16_OUT_S64,
                &SPV_F16_OUT_ALIGNED_S64,
                if state.rows_order {
                    &[]
                } else {
                    &SPV_F16_OUT_RAST_S64
                },
                64,
            )
        } else if state.ctx.can_require_subgroup_size(32) {
            (&SPV_F16_OUT_S32, &SPV_F16_OUT_ALIGNED_S32, &[], 32)
        } else {
            set_error("fp16-output matmul: the device can require neither 64 nor 32 lanes");
            return Err(ACK_ERR_UNSUPPORTED_OP);
        };
    let mut built: Vec<Kernel> = Vec::with_capacity(RAST_AT + 4);
    for spv in edge.iter().chain(aligned.iter()).chain(rast.iter()) {
        match state
            .ctx
            .kernel_with_subgroup(spv, 3, COOPMAT_PUSH_BYTES as u32, Some(width))
        {
            Ok(k) => built.push(k),
            Err(e) => {
                for k in built.drain(..) {
                    state.ctx.destroy_kernel(k);
                }
                set_error(format!("fp16-output matmul pipeline ({width} lanes): {e}"));
                return Err(code_for(&e));
            }
        }
    }
    state.matmul_f16_out = Some(built);
    Ok(())
}

/// `C = A . B` with fp16 operands and an FP16 `C` (2026-09-30): the fp16
/// twins of the bf16-output modules (the aligned ones for a shape on the
/// 128 x 128 x 32 tile, the edge ones otherwise, as `ack_matmul_f16` takes its
/// own), so the f32 accumulator is rounded in the product's own epilogue by
/// convert/v1's f32 -> f16 (to nearest even, NaN -> sign | 0x7E00): bit for
/// bit what `ack_matmul_f16` followed by convert/v1 writes, without the f32
/// block or the convert dispatch. `n` must be even: C is written in fp16
/// pairs, and a pair must not straddle two rows. A device without fp16
/// cooperative matrices is refused with `ACK_ERR_UNSUPPORTED_OP` before its
/// buffers are looked at. RECORDED like every operator entry.
///
/// OPTIONAL, resolved by name and in no pointer table, like
/// `ack_matmul_bf16_to_bf16`: `ACK_ABI_VERSION` does not move, and a caller
/// treats its absence as "convert after the product".
/// # Safety
/// Every handle must come from this device; `ms_out` must be null or point to
/// a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_f16_to_f16(
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
    unsafe { no_duration(ms_out) };
    if m == 0 || n == 0 || k == 0 {
        set_error(format!(
            "shape {m}x{n}x{k} has a zero dimension: the dispatch grid would be empty"
        ));
        return ACK_ERR_SHAPE;
    }
    if n % 2 != 0 {
        set_error(format!(
            "matmul-f16-out-odd-n: n = {n}; C is written in fp16 pairs, which an odd row would straddle"
        ));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = matmul_f16_out_kernels(&mut guard) {
        return code;
    }
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
    let need_a = m as u64 * k as u64 * 2;
    let need_b = k as u64 * n as u64 * 2;
    let need_c = m as u64 * n as u64 * 2;
    if a.bytes < need_a || b.bytes < need_b || c.bytes < need_c {
        set_error(format!(
            "buffers too small for {m}x{n}x{k}: A {} (need {need_a}), B {} (need {need_b}), C {} (need {need_c})",
            a.bytes, b.bytes, c.bytes
        ));
        return ACK_ERR_SIZE;
    }
    if c.id() == a.id() || c.id() == b.id() {
        set_error("matmul-f16-out-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let set = guard.matmul_f16_out.as_ref().expect("built above");
    let aligned = bf16_out_aligned(m, n, k);
    let index = rast_index(aligned, n, set.len(), a_t, b_t)
        .unwrap_or((aligned as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize));
    let kernel = &set[index];
    if let Err(e) = guard.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let push = coopmat_push(m, n, k, 0, 0, 0, 0, 0, 0, 0);
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &push,
        [n.div_ceil(TILE), m.div_ceil(TILE), 1],
        1,
        &access,
    ))
}

/// `ack_matmul_f16_z` with an FP16 `C` (2026-09-30): its batches (zmode 0)
/// through the fp16-output modules, so each product's f32 accumulator is
/// rounded in its own epilogue by convert/v1's f32 -> f16 -- bit for bit what
/// `ack_matmul_f16_z` followed by convert/v1 writes, without the f32 block or
/// the convert dispatch. A K-split (zmode 1) is refused by name
/// (`matmul-f16-out-k-split`): its partials are f32 sums the caller reduces.
/// `n` and `zc` must be even (`matmul-f16-out-odd-n`): C is written in fp16
/// pairs, and a pair must straddle neither two rows nor two batches. Every
/// other rule -- the z grid, the strides, the pitches, the refusals and the
/// module choice -- is `ack_matmul_f16_z`'s, with C counted in 2-byte
/// elements. OPTIONAL, resolved by name, like `ack_matmul_f16_to_f16`.
/// # Safety
/// Every handle must come from this device; `ms_out` must be null or point to
/// a writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_f16_z_to_f16(
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
    unsafe { no_duration(ms_out) };
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
    if zmode != 0 || kz != 0 {
        set_error(format!(
            "matmul-f16-out-k-split: zmode {zmode}, kz {kz}; the fp16-output modules write batches (zmode 0) only, a K-split's partials are f32 sums the caller reduces"
        ));
        return ACK_ERR_SHAPE;
    }
    if n % 2 != 0 || zc % 2 != 0 {
        set_error(format!(
            "matmul-f16-out-odd-n: n = {n}, zc = {zc}; C is written in fp16 pairs, which an odd row or batch stride would straddle"
        ));
        return ACK_ERR_SHAPE;
    }
    let mn = u64::from(m) * u64::from(n);
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
    if z > 1 && (za < a_cols || zb < b_cols || u64::from(zc) < mn) {
        set_error(format!(
            "z strides {za}/{zb}/{zc} below the row widths {a_cols}/{b_cols} or the {m}x{n} output block"
        ));
        return ACK_ERR_SHAPE;
    }
    // `ack_matmul_f16_z`'s module choice, zmode 0
    let edge = m % TILE != 0
        || n % TILE != 0
        || k % K_MULTIPLE != 0
        || lda % Z_STRIDE_MULTIPLE != 0
        || ldb % Z_STRIDE_MULTIPLE != 0
        || za % Z_STRIDE_MULTIPLE != 0
        || zb % Z_STRIDE_MULTIPLE != 0;
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if let Err(code) = matmul_f16_out_kernels(&mut guard) {
        return code;
    }
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
        set_error("matmul-f16-out-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let last = u64::from(z - 1);
    let a_span = u64::from(a_rows - 1) * u64::from(lda) + u64::from(a_cols);
    let b_span = u64::from(b_rows - 1) * u64::from(ldb) + u64::from(b_cols);
    let need_a = (last * u64::from(za) + a_span) * 2;
    let need_b = (last * u64::from(zb) + b_span) * 2;
    let need_c = (last * u64::from(zc) + mn) * 2;
    if a.bytes < need_a || b.bytes < need_b || c.bytes < need_c {
        set_error(format!(
            "buffers too small for {z} x {m}x{n}x{k}: A {} (need {need_a}), B {} (need {need_b}), C {} (need {need_c})",
            a.bytes, b.bytes, c.bytes
        ));
        return ACK_ERR_SIZE;
    }
    // the fp16-output set is indexed aligned*4 + a_t*2 + b_t, or by `rast_index`
    let set = guard.matmul_f16_out.as_ref().expect("built above");
    let index = rast_index(!edge, n, set.len(), a_t, b_t)
        .unwrap_or(((!edge) as usize) * 4 + ((a_t != 0) as usize) * 2 + ((b_t != 0) as usize));
    let kernel = &set[index];
    if let Err(e) = guard.ctx.bind(kernel, &[a, b, c]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let push = coopmat_push(m, n, k, 0, za, zb, zc, 0, lda, ldb);
    let access = [
        Access::read(a.id()),
        Access::read(b.id()),
        Access::write(c.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &push,
        [n.div_ceil(TILE), m.div_ceil(TILE), z],
        1,
        &access,
    ))
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
    if let Err(code) = ensure_pointwise_modules(
        &mut guard,
        u32::from_le_bytes([block[12], block[13], block[14], block[15]]),
    ) {
        return code;
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
    let rc = record_pointwise(&guard, &plan, x, y, z, out);
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// Build the pointwise modules a block's storage `flags` select, if this device
/// has not built them yet (`ack_pointwise`'s first half, shared with the
/// families that record the same operations: src/moe_backward.rs). 0 is the
/// f32 module every device builds at open; the bf16 and bool ones are built on
/// the first call that asks for one, so a device that never sees such an
/// operand never builds them, and the vector module for the same storage with
/// them. It runs before any buffer is resolved because the guard cannot be
/// borrowed mutably while they are alive. A slot with two storage bits, or a
/// combination no module is built for, is refused by name. On a refusal the
/// message is set and the code returned.
pub(crate) fn ensure_pointwise_modules(state: &mut Inner, flags: u32) -> Result<(), i32> {
    let module = match pointwise_ops::storage_module(flags) {
        Some(module) => module,
        None => {
            set_error("pointwise: plan refused: pointwise-storage-unbuilt or -conflict");
            return Err(ACK_ERR_SHAPE);
        }
    };
    if pointwise_ops::module_has_bool(module) && !state.ctx.bool_storage {
        set_error(
            "pointwise: bool storage needs storageBuffer8BitAccess and shaderInt8,              which this device did not enable",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    if module != 0 && state.pointwise_storage[module - 1].is_none() {
        let built = state.ctx.kernel(
            SPV_POINTWISE_STORAGE[module - 1],
            pointwise_ops::OPERANDS as u32,
            pointwise_ops::PUSH_BYTES as u32,
        );
        match built {
            Ok(k) => state.pointwise_storage[module - 1] = Some(k),
            Err(e) => {
                set_error(format!("pointwise storage pipeline {module}: {e}"));
                return Err(code_for(&e));
            }
        }
    }
    // The vector module for the same storage. Whether a call uses it is the
    // plan's to decide (`record_pointwise`).
    if let Some(vector) = pointwise_ops::vector_module(module) {
        if !state.pointwise_vector_off && state.pointwise_vector[vector].is_none() {
            let built = state.ctx.kernel(
                SPV_POINTWISE_VECTOR[vector],
                pointwise_ops::OPERANDS as u32,
                pointwise_ops::PUSH_BYTES as u32,
            );
            match built {
                Ok(k) => state.pointwise_vector[vector] = Some(k),
                Err(e) => {
                    set_error(format!("pointwise vector pipeline {vector}: {e}"));
                    return Err(code_for(&e));
                }
            }
        }
    }
    Ok(())
}

/// Record one planned pointwise dispatch on the module its plan selects
/// (`ack_pointwise`'s second half, shared as `ensure_pointwise_modules` is):
/// the vector module when the plan is eligible and the device has not turned
/// vectors off, else the plan's storage module. `ensure_pointwise_modules`
/// must have run for the plan's flags. Returns `ACK_OK` once RECORDED.
pub(crate) fn record_pointwise(
    state: &Inner,
    plan: &pointwise_ops::PointwisePlan,
    x: &Buffer,
    y: &Buffer,
    z: &Buffer,
    out: &Buffer,
) -> i32 {
    // Defensive: ack_open fails when the f32 pipeline cannot be built, so an
    // open device always carries it; the branch names the state rather than
    // trusting it.
    // the plan's own reading of the same bits; it refused the block if they
    // were not ones this version implements, so the two agree by construction
    let selected = plan.storage_module();
    let vector = pointwise_ops::vector_module(selected)
        .filter(|_| !state.pointwise_vector_off && plan.vector_eligible());
    let slot = if let Some(vector) = vector {
        state.pointwise_vector[vector].as_ref()
    } else if selected == 0 {
        state.pointwise.as_ref()
    } else {
        state.pointwise_storage[selected - 1].as_ref()
    };
    let kernel = match slot {
        Some(k) => k,
        None => {
            set_error("pointwise: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = state.ctx.bind(kernel, &[x, y, z, out]) {
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
    // the coalesced block (2026-09-27): the same elements and addresses, in
    // as few dimensions as the operands allow
    recorded(state.ctx.dispatch(
        kernel,
        &plan.dispatch_push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
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
            3 => ReduceOp::AbsMax,
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
    // reduce-f32/v3's f32 module, built on the first reduction of the device
    // rather than at open (the plan chooses v3 for most reductions of the AKV
    // pass, so a device that reduces at all builds it once; one that never
    // reduces builds nothing). Not on the v1-only arm, which never selects it.
    if let Err(code) = ensure_reduce_v3_f32(&mut guard) {
        return code;
    }
    if storage & reduce_ops::FLAG_F64 != 0 && guard.reduce_f64.is_none() {
        let state: &mut Inner = &mut guard;
        match state
            .ctx
            .kernel(SPV_REDUCE_F64, 2, reduce_ops::PUSH_BYTES as u32)
        {
            Ok(k) => state.reduce_f64 = Some(k),
            Err(e) => {
                set_error(format!("reduce f64 pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    if storage & (reduce_ops::FLAG_IN_BF16 | reduce_ops::FLAG_OUT_BF16) != 0 {
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
                (
                    &mut state.reduce_v3[index + 1],
                    SPV_REDUCE_V3[index + 1],
                    reduce_ops::PUSH_BYTES_V3 as u32,
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
    let in_bytes = reduce_ops::storage_element_bytes(storage, true);
    let out_bytes = reduce_ops::storage_element_bytes(storage, false);
    // partials are f32, or float64 under FLAG_F64
    let partial_bytes = if storage & reduce_ops::FLAG_F64 != 0 {
        8
    } else {
        4
    };
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
    // A two-pass plan writes f32 PARTIALS, `plan.first().output_len()` of
    // them. The caller's `partials` serve when they are at least that large;
    // otherwise (null, or smaller than this plan's split law asks) the kit
    // allocates its own for this call and retires it into the batch at once,
    // so the pool takes it back after the batch that reads it completes.
    // 2026-09-26: the split law became a function of the view's geometry
    // (reduce-f32/v3 and the occupancy splits), which the torch adapter no
    // longer mirrors, so an under-sized or absent caller scratch is no longer
    // a refusal. A caller scratch that aliases the input or output still is.
    if let Some(p) = partials {
        if p.id() == x.id() || p.id() == out.id() {
            set_error("reduce: plan refused: reduce-partials-alias");
            return ACK_ERR_SHAPE;
        }
    }
    let need = plan.second().map(|_| plan.first().output_len() as u64);
    let supplied = match (need, partials) {
        (Some(n), Some(p)) if p.bytes / partial_bytes >= n => Some(p),
        _ => None,
    };
    let internal = match (need, supplied) {
        (Some(n), None) => match guard.ctx.buffer(n * partial_bytes) {
            Ok(b) => Some(b),
            Err(e) => {
                set_error(format!("reduce: partials of {n} elements: {e}"));
                return code_for(&e);
            }
        },
        _ => None,
    };
    let scratch = supplied.or(internal.as_ref());
    let rc = record_reduce(&guard, &plan, x, out, scratch);
    // retired, not destroyed: the batch that reads it may still be open
    if let Some(b) = internal {
        guard.ctx.destroy_buffer(b);
    }
    if rc == ACK_OK {
        // recorded, not run: no duration exists for this call to report
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// reduce-f32/v3's f32 module, built on the first reduction of the device
/// rather than at open (`ack_reduce`'s, shared with src/moe_backward.rs). Not
/// on the v1-only arm, which never selects it. On a refusal the message is set
/// and the code returned.
pub(crate) fn ensure_reduce_v3_f32(state: &mut Inner) -> Result<(), i32> {
    if !state.reduce_v1_only && state.reduce_v3[0].is_none() {
        match state
            .ctx
            .kernel(SPV_REDUCE_V3[0], 2, reduce_ops::PUSH_BYTES_V3 as u32)
        {
            Ok(k) => state.reduce_v3[0] = Some(k),
            Err(e) => {
                set_error(format!("reduce_v3 pipeline: {e}"));
                return Err(code_for(&e));
            }
        }
    }
    Ok(())
}

/// Whether this device's reduce plans may choose v2 and v3: false on the
/// v1-only arm the device took at open (`ACK_REDUCE_V1_ONLY`), which every
/// `ReducePlan::with_storage` built for it must be told.
pub(crate) fn reduce_beyond_v1(state: &Inner) -> bool {
    !state.reduce_v1_only
}

/// Record a reduce plan's one or two passes. `scratch` is the partials
/// buffer when the plan has a second pass (the caller's, or the kit's own).
pub(crate) fn record_reduce(
    guard: &Inner,
    plan: &ReducePlan,
    x: &Buffer,
    out: &Buffer,
    scratch: Option<&Buffer>,
) -> i32 {
    // Defensive: ack_open fails when the pipeline cannot be built, so an
    // open device always carries the v1 and v2 kernels, and `ack_reduce`
    // built v3's f32 module before planning; the branches name the state
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
    // across the lanes, v3 gives a contiguous reduced span's rows a few lanes
    // each (and takes every second pass it can) -- and then the STORAGE
    // MODULE that pass's own two bits name. Module 0 of v1 and v2 is the f32
    // pair built at open; the rest were built on demand above.
    let module_of = |pass: &reduce_ops::ReducePass| match (pass.kernel(), pass.storage_module()) {
        (reduce_ops::ReduceKernel::V1, 4) => guard.reduce_f64.as_ref(),
        (_, 4) => None,
        (reduce_ops::ReduceKernel::V1, 0) => Some(kernel_v1),
        (reduce_ops::ReduceKernel::V2, 0) => Some(kernel_v2),
        (reduce_ops::ReduceKernel::V1, i) => guard.reduce_storage[i - 1].as_ref(),
        (reduce_ops::ReduceKernel::V2, i) => guard.reduce_v2_storage[i - 1].as_ref(),
        (reduce_ops::ReduceKernel::V3, i) => guard.reduce_v3[i].as_ref(),
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
    crate::vq4_ops::SCHEMA
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
/// u32 words. See vq4_ops and vq4.comp for the closed per-operation layout.
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
    let mut capacities = [0u64; crate::vq4_ops::OPERANDS];
    let mut ids = [0u64; crate::vq4_ops::OPERANDS];
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
        match guard.ctx.kernel(
            SPV_VQ,
            crate::vq4_ops::OPERANDS as u32,
            VQ_PUSH_BYTES as u32,
        ) {
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
    unsafe {
        loss_entry(
            dev,
            primary,
            targets,
            upstream,
            out,
            rows,
            cols,
            op,
            reduction,
            ignore_index,
            target_values,
            targets_len,
            uploaded_bytes,
            ms_out,
            0,
        )
    }
}

/// `ack_loss` over bf16 storage (2026-09-28), for a caller whose logits are bf16
/// (autocast's head output). `flags` is `ACK_LOSS_PRIMARY_BF16` (the forward or
/// the backward reads the logits as bf16, each widened exactly: the values the
/// f32 entry would read after the caller's own upcast) optionally with
/// `ACK_LOSS_OUTPUT_BF16` (the backward writes its gradient as bf16, rounded to
/// nearest even as the cast kernel rounds: the values the f32 entry writes, cast).
/// The arithmetic is the f32 module's own source, so a caller drops the upcast
/// and the cast and keeps the numbers. Refused by name: an unknown flag
/// (`loss-bf16-unknown-flag`), bf16 logits for any other operation
/// (`loss-bf16-logits-op`), a bf16 gradient outside the backward or without bf16
/// logits (`loss-bf16-output-op`), and an odd `cols` with a bf16 gradient
/// (`loss-bf16-output-odd-cols`: it is written in pairs). Every other rule is
/// `ack_loss`'s, with the bf16 buffers' capacities counted in 2-byte elements.
///
/// OPTIONAL, resolved by name and in no pointer table, like
/// `ack_matmul_bf16_to_bf16`: `ACK_ABI_VERSION` does not move.
/// # Safety
/// As `ack_loss`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_loss_bf16(
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
    flags: u32,
) -> i32 {
    unsafe {
        loss_entry(
            dev,
            primary,
            targets,
            upstream,
            out,
            rows,
            cols,
            op,
            reduction,
            ignore_index,
            target_values,
            targets_len,
            uploaded_bytes,
            ms_out,
            flags,
        )
    }
}

fn loss_bf16_kernels(state: &mut Inner) -> Result<(), i32> {
    if state.loss_bf16.is_some() {
        return Ok(());
    }
    let mut built: Vec<Kernel> = Vec::with_capacity(SPV_LOSS_BF16.len());
    for spv in SPV_LOSS_BF16.iter() {
        match state.ctx.kernel(spv, 4, LOSS_PUSH_BYTES as u32) {
            Ok(k) => built.push(k),
            Err(e) => {
                for k in built.drain(..) {
                    state.ctx.destroy_kernel(k);
                }
                set_error(format!("loss (bf16 storage) pipeline: {e}"));
                return Err(code_for(&e));
            }
        }
    }
    state.loss_bf16 = Some(built);
    Ok(())
}

/// The rotate-half rotary position embedding (op 0) and its backward (op 1)
/// over dense FP32 rows (rotary-f32/v1, 2026-09-29).
///
/// `x` holds `rows` rows of `cols` (even) elements: the input for the forward,
/// the upstream gradient of the output for the backward. Row `r` sits at
/// position `(r / pos_div) % positions`. `cos` and `sin` each hold `positions`
/// rows of `cols`. With `h = cols / 2`:
/// - the forward writes `x * cos + rot(x) * sin`, where `rot(x)` is
///   `cat(-x[h:], x[:h])`;
/// - the backward writes that law's gradient: `g * cos` plus the rotated
///   slices of `g * sin`.
///
/// Every product is rounded, then their sum, in the order torch's
/// decomposition and its autograd use. So the values are that decomposition's,
/// bit for bit, except the sign of an exact zero in the backward.
///
/// Refused by name, before any device lookup: a zero dimension
/// (`rotary-dimension-out-of-range`), an odd `cols` (`rotary-odd-columns`), an
/// element count past the family's bound (`rotary-element-limit-exceeded`),
/// and an unknown operation (`rotary-operation-out-of-range`). After the
/// lookup: a buffer smaller than its role (`rotary-<role>-too-small`), and an
/// output that is one of the inputs (`rotary-output-aliases-input`).
///
/// OPTIONAL, resolved by name and in no pointer table, like `ack_loss_bf16`:
/// `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_rotary(
    dev: *const AckDevice,
    x: *const Buffer,
    cos: *const Buffer,
    sin: *const Buffer,
    out: *const Buffer,
    rows: u32,
    cols: u32,
    op: u32,
    pos_div: u32,
    positions: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let op = match RotaryOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("rotary: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match RotaryPlan::new(rows, cols, op, pos_div, positions) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "rotary: {rows} rows of {cols} at {positions} positions (divisor {pos_div}): plan refused: {e}"
            ));
            return match e {
                crate::rotary_ops::RotaryPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.rotary.is_none() {
        match guard
            .ctx
            .kernel(SPV_ROTARY, 4, crate::rotary_ops::PUSH_BYTES as u32)
        {
            Ok(k) => guard.rotary = Some(k),
            Err(e) => {
                set_error(format!("rotary pipeline: {e}"));
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
    let cos = match buffer_of(&device, &guard, cos) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let sin = match buffer_of(&device, &guard, sin) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    for (buffer, needed, name) in [
        (x, plan.elements(), "primary"),
        (cos, plan.table_len(), "cos"),
        (sin, plan.table_len(), "sin"),
        (out, plan.elements(), "output"),
    ] {
        if buffer.bytes / 4 < needed as u64 {
            set_error(format!(
                "rotary: {name} of {} elements for {needed}: plan refused: rotary-{name}-too-small",
                buffer.bytes / 4
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes the output while other invocations still read the
    // inputs, so the output may be none of them
    if out.id() == x.id() || out.id() == cos.id() || out.id() == sin.id() {
        set_error("rotary: plan refused: rotary-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.rotary.as_ref() {
        Some(k) => k,
        None => {
            set_error("rotary: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[x, cos, sin, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(x.id()),
        Access::read(cos.id()),
        Access::read(sin.id()),
        Access::write(out.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// The SiLU-gated activation of one packed projection (op 0) and its backward
/// (op 1), over dense rows of FP32 (`storage` 0) or bf16 (`storage` 1)
/// (swiglu/v1, 2026-09-29).
///
/// `packed` holds `rows` rows of `2 * cols` elements: `u` in the first `cols`,
/// the gate `g` in the second.
/// - The forward writes `u * silu(g)`, `rows` rows of `cols`, to `out`.
///   `upstream` is not read, and may be null.
/// - The backward reads `upstream`, the gradient of that output (`rows` rows
///   of `cols`), and writes the gradient of `packed` (`rows` rows of
///   `2 * cols`) to `out`: both halves, in one dispatch.
///
/// Every intermediate is rounded to the storage type where torch's
/// decomposition (`silu`, `mul` and their autograd) rounds its tensors, in
/// that decomposition's order and with the pointwise family's arithmetic. So
/// on one device every value that is not a NaN is that decomposition's through
/// the pointwise family (operations 14, 2 and 37), bit for bit. A NaN is a NaN
/// in both, with its sign and payload free: a multiply of two NaNs returns one of them, and which one follows the operand order each compilation gives it.
///
/// Refused by name, before any device lookup: a zero dimension
/// (`swiglu-dimension-out-of-range`), a packed operand past the family's bound
/// (`swiglu-element-limit-exceeded`), an unknown operation
/// (`swiglu-operation-out-of-range`) and an unknown storage
/// (`swiglu-storage-out-of-range`). After the lookup: a buffer smaller than its
/// role (`swiglu-<role>-too-small`), and an output that is one of the inputs
/// (`swiglu-output-aliases-input`).
///
/// OPTIONAL, resolved by name and in no pointer table, like `ack_rotary`:
/// `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_swiglu(
    dev: *const AckDevice,
    packed: *const Buffer,
    upstream: *const Buffer,
    out: *const Buffer,
    rows: u32,
    cols: u32,
    op: u32,
    storage: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let (op, storage) = match (SwigluOp::try_from(op), SwigluStorage::try_from(storage)) {
        (Ok(op), Ok(storage)) => (op, storage),
        (Err(e), _) | (_, Err(e)) => {
            set_error(format!("swiglu: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match SwigluPlan::new(rows, cols, op, storage) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "swiglu: {rows} rows of 2 x {cols}: plan refused: {e}"
            ));
            return match e {
                SwigluPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let module = storage as usize;
    if guard.swiglu[module].is_none() {
        match guard
            .ctx
            .kernel(SPV_SWIGLU[module], 3, crate::swiglu_ops::PUSH_BYTES as u32)
        {
            Ok(k) => guard.swiglu[module] = Some(k),
            Err(e) => {
                set_error(format!("swiglu pipeline ({storage:?}): {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle; the forward reads no upstream
    let packed = match buffer_of(&device, &guard, packed) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let upstream = match op {
        SwigluOp::Forward => None,
        SwigluOp::Backward => match buffer_of(&device, &guard, upstream) {
            Ok(b) => Some(b),
            Err(code) => return code,
        },
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let mut roles = vec![(packed, plan.packed_len(), "packed")];
    if let Some(upstream) = upstream {
        roles.push((upstream, plan.elements(), "upstream"));
    }
    roles.push((out, plan.output_len(), "output"));
    for (buffer, needed, name) in roles {
        if buffer.bytes / storage.bytes() < needed as u64 {
            set_error(format!(
                "swiglu: {name} of {} elements for {needed}: plan refused: swiglu-{name}-too-small",
                buffer.bytes / storage.bytes()
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes the output while other invocations still read the
    // inputs, so the output may be none of them
    if out.id() == packed.id() || upstream.is_some_and(|u| out.id() == u.id()) {
        set_error("swiglu: plan refused: swiglu-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.swiglu[module].as_ref() {
        Some(k) => k,
        None => {
            set_error("swiglu: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    // the forward binds the packed operand in the upstream's slot, which it never reads
    let upstream = upstream.unwrap_or(packed);
    if let Err(e) = guard.ctx.bind(kernel, &[packed, upstream, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // one declaration a bound slot, in order: the forward's packed operand twice
    let access = [
        Access::read(packed.id()),
        Access::read(upstream.id()),
        Access::write(out.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// The weighted sum of gathered table rows (op 0), the weights' gradient (op 1) and the table's gradient (op 2),
/// FP32 (weighted-bag/v1, 2026-10-22; `crate::weighted_bag_ops`).
///
/// `table` holds `vocab` rows of `d`; `idx` (u32) and `w` hold `rows` rows of `k`.
/// - op 0 writes `out[r] = sum_s w[r, s] * table[idx[r, s]]`, `rows` rows of `d`. `up` and `scratch` are not read.
/// - op 1 reads `up` (`rows` rows of `d`, the gradient of op 0's output) and writes `rows * k` values,
///   `up[r] . table[idx[r, s]]`. `scratch` is not read.
/// - op 2 reads `up` and writes the table's gradient, `vocab` rows of `d` (zeros where no pair names a row), using
///   `scratch` (`3 (vocab + 1) + rows k` u32 words) for its device-built index; `table` is not read.
///
/// Refused by name before any device lookup: a zero dimension (`weighted-bag-dimension-out-of-range`), a geometry
/// past the family's bounds (`weighted-bag-element-limit-exceeded`), an unknown operation
/// (`weighted-bag-operation-out-of-range`). After the lookup: a buffer smaller than its role
/// (`weighted-bag-<role>-too-small`) and an output that is an input or the scratch
/// (`weighted-bag-output-aliases-input`). Every index must be below `vocab`; the caller checks.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null where unread; `ms_out` is null or points to one
/// writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_weighted_bag(
    dev: *const AckDevice,
    table: *const Buffer,
    idx: *const Buffer,
    w: *const Buffer,
    up: *const Buffer,
    out: *const Buffer,
    scratch: *const Buffer,
    rows: u32,
    k: u32,
    vocab: u32,
    d: u32,
    op: u32,
    ms_out: *mut f64,
) -> i32 {
    use crate::weighted_bag_ops::{record, WeightedBagBuffers, WeightedBagOp, WeightedBagPlan};
    unsafe { no_duration(ms_out) };
    let op = match WeightedBagOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("weighted bag: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match WeightedBagPlan::new(rows, k, vocab, d, op) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!(
                "weighted bag: {rows} x {k} of a {vocab} x {d} table: plan refused: {e}"
            ));
            return if e.contains("limit") {
                ACK_ERR_SIZE
            } else {
                ACK_ERR_SHAPE
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.weighted_bag.is_none() {
        match guard.ctx.kernel(
            SPV_WEIGHTED_BAG,
            6,
            crate::weighted_bag_ops::PUSH_BYTES as u32,
        ) {
            Ok(kern) => guard.weighted_bag = Some(kern),
            Err(e) => {
                set_error(format!("weighted bag pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    let reads_table = op != WeightedBagOp::GradTable;
    let reads_up = op != WeightedBagOp::Forward;
    let uses_scratch = op == WeightedBagOp::GradTable;
    let idx = match buffer_of(&device, &guard, idx) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let w = match buffer_of(&device, &guard, w) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    // an unread role binds `w` in its slot, which the kernel never touches through it
    let table = if reads_table {
        match buffer_of(&device, &guard, table) {
            Ok(b) => b,
            Err(code) => return code,
        }
    } else {
        w
    };
    let up = if reads_up {
        match buffer_of(&device, &guard, up) {
            Ok(b) => b,
            Err(code) => return code,
        }
    } else {
        w
    };
    let scratch = if uses_scratch {
        match buffer_of(&device, &guard, scratch) {
            Ok(b) => b,
            Err(code) => return code,
        }
    } else {
        w
    };
    let mut roles = vec![
        (idx, plan.pairs(), "idx"),
        (w, plan.pairs(), "weights"),
        (out, plan.output_words(), "output"),
    ];
    if reads_table {
        roles.push((table, plan.table_words(), "table"));
    }
    if reads_up {
        roles.push((up, plan.rows_words(), "upstream"));
    }
    if uses_scratch {
        roles.push((scratch, plan.scratch_words(), "scratch"));
    }
    for (buffer, needed, name) in roles {
        if buffer.bytes / 4 < needed {
            set_error(format!(
                "weighted bag: {name} of {} words for {needed}: plan refused: weighted-bag-{name}-too-small",
                buffer.bytes / 4
            ));
            return ACK_ERR_SIZE;
        }
    }
    if [table.id(), idx.id(), w.id(), up.id(), scratch.id()].contains(&out.id())
        || (uses_scratch && [table.id(), idx.id(), w.id(), up.id()].contains(&scratch.id()))
    {
        set_error("weighted bag: plan refused: weighted-bag-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.weighted_bag.as_ref() {
        Some(k) => k,
        None => {
            set_error("weighted bag: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    let bufs = WeightedBagBuffers {
        table,
        idx,
        w,
        up,
        out,
        scratch,
    };
    recorded(record(&guard.ctx, kernel, &plan, &bufs))
}

/// The sub-bit student's experts' activation, `out = silu(gate) * (up * g)`
/// (op 0), and its backward (op 1), over dense rows of FP32 (`storage` 0), bf16
/// (1) or fp16 (2) (swiglu-scaled/v1, 2026-09-30).
/// `gate` and `up` hold `rows` rows of `cols`; `scale` holds one row of `cols`
/// for every `width` consecutive rows (`rows` a multiple of `width`).
/// - The forward writes `out`, `rows` rows of `cols`. `upstream`, `out_up`
///   and `out_t` are neither read nor written, and may be null.
/// - The backward reads `upstream`, the gradient of that output, and writes
///   three outputs of `rows` x `cols`, each to its own buffer: gate's gradient
///   to `out`, up's gradient to `out_up`, and `grad_b * up` to `out_t`, whose
///   sum over each group's rows is g's gradient (the caller forms that sum, as
///   autograd's `sum_to_size` does). Separate buffers, so each gradient is
///   read from its buffer's front by the kit's products, which stage an
///   operand at an offset through a copy.
///
/// Every intermediate is rounded to the storage type where torch's
/// decomposition (`silu`, two `mul`s and their autograd) rounds its tensors, in
/// that decomposition's order and with the pointwise family's arithmetic; an
/// fp16 step widens exactly and narrows with convert/v1's integer rounding, as
/// the kit's composed fp16 operations do. So on one device every value that is
/// not a NaN is that decomposition's, bit for bit. A NaN is a NaN in both, with
/// its sign and payload free.
///
/// Refused by name, before any device lookup: a zero dimension or a `width`
/// that does not divide `rows` (`swiglu-dimension-out-of-range`), an output
/// past the family's bound (`swiglu-element-limit-exceeded`), an unknown
/// operation or storage. After the lookup: a buffer smaller than its role
/// (`swiglu-<role>-too-small`), an output that is one of the inputs
/// (`swiglu-output-aliases-input`), and two outputs of the backward that are
/// one buffer (`swiglu-output-aliases-output`).
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_swiglu_scaled(
    dev: *const AckDevice,
    gate: *const Buffer,
    up: *const Buffer,
    scale: *const Buffer,
    upstream: *const Buffer,
    out: *const Buffer,
    out_up: *const Buffer,
    out_t: *const Buffer,
    rows: u32,
    cols: u32,
    width: u32,
    op: u32,
    storage: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let (op, storage) = match (SwigluOp::try_from(op), ScaledStorage::try_from(storage)) {
        (Ok(op), Ok(storage)) => (op, storage),
        (Err(e), _) | (_, Err(e)) => {
            set_error(format!("swiglu-scaled: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match SwigluScaledPlan::new(rows, cols, width, op, storage) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "swiglu-scaled: {rows} rows of {cols}, a scale row a {width}: plan refused: {e}"
            ));
            return match e {
                SwigluPlanError::ElementLimitExceeded => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let module = storage as usize;
    if guard.swiglu_scaled[module].is_none() {
        match guard.ctx.kernel(
            SPV_SWIGLU_SCALED[module],
            7,
            crate::swiglu_ops::SCALED_PUSH_BYTES as u32,
        ) {
            Ok(k) => guard.swiglu_scaled[module] = Some(k),
            Err(e) => {
                set_error(format!("swiglu-scaled pipeline ({storage:?}): {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle; the forward reads no upstream and
    // writes one output
    let gate = match buffer_of(&device, &guard, gate) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let up = match buffer_of(&device, &guard, up) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let scale = match buffer_of(&device, &guard, scale) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let backward = |handle: *const Buffer| match op {
        SwigluOp::Forward => Ok(None),
        SwigluOp::Backward => buffer_of(&device, &guard, handle).map(Some),
    };
    let upstream = match backward(upstream) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out_up = match backward(out_up) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out_t = match backward(out_t) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let mut roles = vec![
        (gate, plan.elements(), "gate"),
        (up, plan.elements(), "up"),
        (scale, plan.scale_len(), "scale"),
    ];
    for (b, name) in [
        (upstream, "upstream"),
        (Some(out), "output"),
        (out_up, "up-grad"),
        (out_t, "scale-term"),
    ] {
        if let Some(b) = b {
            roles.push((b, plan.elements(), name));
        }
    }
    for (buffer, needed, name) in roles {
        if buffer.bytes / storage.bytes() < needed as u64 {
            set_error(format!(
                "swiglu-scaled: {name} of {} elements for {needed}: plan refused: swiglu-{name}-too-small",
                buffer.bytes / storage.bytes()
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes the output while other invocations still read the
    // inputs, so the output may be none of them
    let inputs = [Some(gate), Some(up), Some(scale), upstream];
    let outputs = [Some(out), out_up, out_t];
    for o in outputs.iter().flatten() {
        if inputs.iter().flatten().any(|b| b.id() == o.id()) {
            set_error("swiglu-scaled: plan refused: swiglu-output-aliases-input");
            return ACK_ERR_SHAPE;
        }
    }
    // each output is written whole, so two of them may not be one buffer
    let ids: Vec<u64> = outputs.iter().flatten().map(|b| b.id()).collect();
    if (1..ids.len()).any(|i| ids[..i].contains(&ids[i])) {
        set_error("swiglu-scaled: plan refused: swiglu-output-aliases-output");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.swiglu_scaled[module].as_ref() {
        Some(k) => k,
        None => {
            set_error("swiglu-scaled: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    // the forward binds gate in the upstream's slot, which it never reads, and
    // its output in the two it never writes
    let upstream = upstream.unwrap_or(gate);
    let (out_up, out_t) = (out_up.unwrap_or(out), out_t.unwrap_or(out));
    if let Err(e) = guard
        .ctx
        .bind(kernel, &[gate, up, scale, upstream, out, out_up, out_t])
    {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // one declaration a bound slot, in order: the forward's gate twice and its
    // output three times
    let access = [
        Access::read(gate.id()),
        Access::read(up.id()),
        Access::read(scale.id()),
        Access::read(upstream.id()),
        Access::write(out.id()),
        Access::write(out_up.id()),
        Access::write(out_t.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// A head-kl-ce-f16/v1 plan refusal's return code: a size bound is ACK_ERR_SIZE,
/// every other refusal ACK_ERR_SHAPE.
fn head_loss_code(e: HeadLossPlanError) -> i32 {
    match e {
        HeadLossPlanError::ElementLimitExceeded | HeadLossPlanError::GridAboveLimit => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// head-kl-ce-f16/v1, the FORWARD (2026-09-29): the sub-bit student's head
/// loss needs, for each of `rows` rows of the fp16 head output `yp` (`rows` x
/// `cols`, dense) and the fp32 per-column scale `h` (`cols`), the log-sum-exp
/// of the logits z = f32(yp) * h over every column, and over the columns not
/// among the row's `k` top ids (`top`, `rows` x `k` int64). Both come out in
/// one pass over `yp`, into `out` (2 `rows` f32): every row's sum over all
/// columns, then every row's sum over the rest. A row with a top id outside
/// [0, cols) gets NaN in both, and the id is never used as an address.
///
/// The logits are the correctly rounded f32 products the reference forms, so
/// the sums differ from torch.logsumexp's only in order and in exp's rounding.
///
/// Refused by name, before any device lookup, through `head_loss_ops`:
/// - a zero dimension (`head-loss-dimension-out-of-range`);
/// - an odd `cols` (`head-loss-odd-columns`);
/// - a width, a top-k or an element count past the family's bounds.
///
/// After the lookup: a buffer smaller than its role
/// (`head-loss-<role>-too-small`), an output that is one of the inputs
/// (`head-loss-output-aliases-input`), and a pipeline this device cannot
/// build (`head-loss-pipeline-unavailable`, ACK_ERR_UNSUPPORTED_OP).
///
/// OPTIONAL, resolved by name and in no pointer table, like `ack_rotary`:
/// `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_head_lse_f16(
    dev: *const AckDevice,
    yp: *const Buffer,
    h: *const Buffer,
    top: *const Buffer,
    out: *const Buffer,
    rows: u32,
    cols: u32,
    k: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let plan = match HeadLsePlan::new(rows, cols, k) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "head_lse: {rows} rows of {cols}, top {k}: plan refused: {e}"
            ));
            return head_loss_code(e);
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.head_lse.is_none() {
        match guard
            .ctx
            .kernel(SPV_HEAD_LSE, 4, crate::head_loss_ops::LSE_PUSH_BYTES as u32)
        {
            Ok(k) => guard.head_lse = Some(k),
            Err(e) => {
                set_error(format!(
                    "head_lse pipeline ({e}): plan refused: head-loss-pipeline-unavailable"
                ));
                return ACK_ERR_UNSUPPORTED_OP;
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let yp = match buffer_of(&device, &guard, yp) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let h = match buffer_of(&device, &guard, h) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let top = match buffer_of(&device, &guard, top) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    for (buffer, needed_bytes, name) in [
        (yp, plan.yp_elements() as u64 * 2, "yp"),
        (h, plan.cols() as u64 * 4, "scale"),
        (top, plan.top_entries() as u64 * 8, "top"),
        (out, plan.out_elements() as u64 * 4, "output"),
    ] {
        if buffer.bytes < needed_bytes {
            set_error(format!(
                "head_lse: {name} of {} bytes for {needed_bytes}: plan refused: head-loss-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes the output while other invocations still read the
    // inputs, so the output may be none of them
    if out.id() == yp.id() || out.id() == h.id() || out.id() == top.id() {
        set_error("head_lse: plan refused: head-loss-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.head_lse.as_ref() {
        Some(k) => k,
        None => {
            set_error("head_lse: plan refused: head-loss-pipeline-unavailable");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[yp, h, top, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(yp.id()),
        Access::read(h.id()),
        Access::read(top.id()),
        Access::write(out.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// head-kl-ce-f16/v1, the BACKWARD (2026-09-29): the gradient of
/// g_kl * KL + g_ce * CE with respect to the fp16 head output `yp` (`rows` x
/// `cols`, dense), written at fp16 into `grad` (the same shape), and each group
/// of `rows_per_group` rows' partial of the gradient with respect to the fp32
/// scale `h`, into `partial` (groups x `cols` f32; the caller sums the groups).
///
/// `row_values` holds 3 `rows` + 1 f32: every row's log-sum-exp over all
/// columns, then c_top = g_kl A + g_ce, then c_nt = c_top - g_kl t / Q, then
/// g_ce. `gkp` holds g_kl p (`rows` x `k` f32), the teacher's top-k
/// probabilities times g_kl, and `targets` the next ids (`rows` int64). Each
/// column's gradient is the reference's (research/subbit_expert/moe/student.py
/// `_kl_ce_block_bwd_f`), in its order, with its rounding:
/// - q c_nt off the top-k;
/// - q c_top - g_kl p on a top-k column (its last repeat's p, as torch's
///   scatter keeps the last);
/// - then -g_ce added at the target.
///
/// q is exp(z - lse) of the same z. d/dyp is that gradient times h, rounded to
/// fp16 to nearest even. d/dh sums yp times the gradient over each group's rows.
///
/// Refused by name as `ack_head_lse_f16` is, and also
/// `head-loss-rows-per-group-out-of-range` (0, or above 64) and
/// `head-loss-grid-above-limit`.
///
/// OPTIONAL, resolved by name: `ACK_ABI_VERSION` does not move.
/// # Safety
/// As `ack_head_lse_f16`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_head_grad_f16(
    dev: *const AckDevice,
    yp: *const Buffer,
    h: *const Buffer,
    top: *const Buffer,
    targets: *const Buffer,
    row_values: *const Buffer,
    gkp: *const Buffer,
    grad: *const Buffer,
    partial: *const Buffer,
    rows: u32,
    cols: u32,
    k: u32,
    rows_per_group: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let plan = match HeadGradPlan::new(rows, cols, k, rows_per_group) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "head_grad: {rows} rows of {cols}, top {k}, {rows_per_group} rows a group: plan refused: {e}"
            ));
            return head_loss_code(e);
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.head_grad.is_none() {
        match guard.ctx.kernel(
            SPV_HEAD_GRAD,
            8,
            crate::head_loss_ops::GRAD_PUSH_BYTES as u32,
        ) {
            Ok(k) => guard.head_grad = Some(k),
            Err(e) => {
                set_error(format!(
                    "head_grad pipeline ({e}): plan refused: head-loss-pipeline-unavailable"
                ));
                return ACK_ERR_UNSUPPORTED_OP;
            }
        }
    }
    let mut bound = Vec::with_capacity(8);
    for handle in [yp, h, top, targets, row_values, gkp, grad, partial] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let (yp, h, top, targets) = (bound[0], bound[1], bound[2], bound[3]);
    let (row_values, gkp, grad, partial) = (bound[4], bound[5], bound[6], bound[7]);
    for (buffer, needed_bytes, name) in [
        (yp, plan.yp_elements() as u64 * 2, "yp"),
        (h, plan.cols() as u64 * 4, "scale"),
        (top, plan.top_entries() as u64 * 8, "top"),
        (targets, plan.rows() as u64 * 8, "targets"),
        (
            row_values,
            plan.row_value_elements() as u64 * 4,
            "row-values",
        ),
        (gkp, plan.top_entries() as u64 * 4, "top-term"),
        (grad, plan.yp_elements() as u64 * 2, "grad"),
        (partial, plan.partial_elements() as u64 * 4, "partial"),
    ] {
        if buffer.bytes < needed_bytes {
            set_error(format!(
                "head_grad: {name} of {} bytes for {needed_bytes}: plan refused: head-loss-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    let inputs = [
        yp.id(),
        h.id(),
        top.id(),
        targets.id(),
        row_values.id(),
        gkp.id(),
    ];
    if grad.id() == partial.id() || inputs.contains(&grad.id()) || inputs.contains(&partial.id()) {
        set_error("head_grad: plan refused: head-loss-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.head_grad.as_ref() {
        Some(k) => k,
        None => {
            set_error("head_grad: plan refused: head-loss-pipeline-unavailable");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard.ctx.bind(
        kernel,
        &[yp, h, top, targets, row_values, gkp, grad, partial],
    ) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(yp.id()),
        Access::read(h.id()),
        Access::read(top.id()),
        Access::read(targets.id()),
        Access::read(row_values.id()),
        Access::read(gkp.id()),
        Access::write(grad.id()),
        Access::write(partial.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// bitsandbytes' NF4 with double quantisation, decoded from its stored form
/// (nf4-decode/v1, 2026-09-30): `n` values of QLoRA's 4-bit NormalFloat into
/// FP32 (`storage` 0), bf16 (1) or fp16 (2), one dispatch.
///
/// The stored form, each buffer read from its front:
/// - `packed`: two 4-bit indices a byte, the first value in the HIGH nibble;
/// - `absmax_q`: one byte a block of 64 values, an index into the map;
/// - `absmax2`: one FP32 value a nested block of 256 blocks;
/// - `tables`: 272 FP32 values, the 16 code values then the 256 map values;
/// - `starts`: `members` int64 values, each member's first value from the
///   source's start, ascending from 0, every one a whole nested block;
/// - `offsets`: `members` FP32 values, each member's offset.
///
/// The source starts `src_base` values into the buffers (a whole nested
/// block: a member view read in place). With `selected` 0 the output is the
/// source, member by member; otherwise `ids` holds `selected` int64 member
/// indices, every member is `member_values` long, and output member `j` is
/// member `ids[j]`, so `n` is `selected * member_values`.
///
/// Value `s` of member `m` in block `b` is `code[index(s)] * fl(fl(map[
/// absmax_q[b]] * absmax2[b / 256]) + offsets[m])`, every product and sum one
/// FP32 rounding with no fused multiply-add, then one rounding to the output
/// storage: the reference decode's arithmetic and bitsandbytes', so every value
/// that is not a NaN is theirs bit for bit, for results in FP32's normal range
/// or zero. A member index or source value past the buffers writes NaN to its
/// values instead of reading (the kernel's own bound: `starts` and `ids` are
/// device data this entry does not read).
///
/// Refused by name before any device lookup: an unknown storage
/// (`nf4-storage-out-of-range`). After the lookup, the plan's names: no values,
/// more than 2^31 (`nf4-zero-values`, `nf4-value-limit-exceeded`), no members
/// (`nf4-members-out-of-range`), a selection whose member size is not a
/// positive multiple of 64 or whose output is not `selected * member_values`
/// (`nf4-selection-out-of-range`), a base off a nested block or past the
/// buffers (`nf4-base-out-of-range`), a whole-source decode longer than the
/// buffers hold (`nf4-source-too-small`); then a buffer smaller than its role
/// (`nf4-<role>-too-small`) and an output that is one of the inputs
/// (`nf4-output-aliases-input`).
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null (`ids` may be
/// null when `selected` is 0); `ms_out` is null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_nf4_decode(
    dev: *const AckDevice,
    packed: *const Buffer,
    absmax_q: *const Buffer,
    absmax2: *const Buffer,
    tables: *const Buffer,
    starts: *const Buffer,
    offsets: *const Buffer,
    ids: *const Buffer,
    out: *const Buffer,
    n: u32,
    members: u32,
    selected: u32,
    member_values: u32,
    src_base: u32,
    storage: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let storage = match Nf4Storage::try_from(storage) {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("nf4-decode: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let module = storage as usize;
    if guard.nf4_decode[module].is_none() {
        match guard
            .ctx
            .kernel(SPV_NF4_DECODE[module], 8, crate::nf4_ops::PUSH_BYTES as u32)
        {
            Ok(k) => guard.nf4_decode[module] = Some(k),
            Err(e) => {
                set_error(format!("nf4-decode pipeline ({storage:?}): {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle; the whole-source decode reads no ids
    let mut bound = Vec::with_capacity(8);
    for handle in [packed, absmax_q, absmax2, tables, starts, offsets] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let ids = if selected == 0 {
        None
    } else {
        match buffer_of(&device, &guard, ids) {
            Ok(b) => Some(b),
            Err(code) => return code,
        }
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let (packed, absmax_q, absmax2, tables, starts, offsets) =
        (bound[0], bound[1], bound[2], bound[3], bound[4], bound[5]);
    let covered = crate::nf4_ops::values_covered(packed.bytes, absmax_q.bytes, absmax2.bytes);
    let plan = match Nf4DecodePlan::new(
        n,
        members,
        selected,
        member_values,
        src_base,
        covered,
        storage,
    ) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "nf4-decode: {n} values, {members} member(s), {selected} selected, base {src_base}, \
                 {covered} covered: plan refused: {e}"
            ));
            return match e {
                Nf4PlanError::ValueLimitExceeded | Nf4PlanError::SourceTooSmall => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let mut roles = vec![
        (tables, 4 * TABLE_ENTRIES as u64, "tables"),
        (starts, 8 * u64::from(members), "starts"),
        (offsets, 4 * u64::from(members), "offsets"),
    ];
    if let Some(ids) = ids {
        roles.push((ids, 8 * u64::from(selected), "ids"));
    }
    roles.push((out, plan.output_bytes(), "output"));
    for (buffer, needed, name) in roles {
        if buffer.bytes < needed {
            set_error(format!(
                "nf4-decode: {name} of {} bytes for {needed}: plan refused: nf4-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes the output while other invocations still read the
    // inputs, so the output may be none of them
    let inputs = [
        Some(packed),
        Some(absmax_q),
        Some(absmax2),
        Some(tables),
        Some(starts),
        Some(offsets),
        ids,
    ];
    if inputs.iter().flatten().any(|b| b.id() == out.id()) {
        set_error("nf4-decode: plan refused: nf4-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.nf4_decode[module].as_ref() {
        Some(k) => k,
        None => {
            set_error("nf4-decode: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    // the whole-source decode binds starts in the ids' slot, which it never reads
    let ids = ids.unwrap_or(starts);
    let slots = [packed, absmax_q, absmax2, tables, starts, offsets, ids, out];
    if let Err(e) = guard.ctx.bind(kernel, &slots) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(packed.id()),
        Access::read(absmax_q.id()),
        Access::read(absmax2.id()),
        Access::read(tables.id()),
        Access::read(starts.id()),
        Access::read(offsets.id()),
        Access::read(ids.id()),
        Access::write(out.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// Build nf4-matmul/v1's pair for one element type on its first call, at the
/// widest subgroup the device can require (64, then 32), as the fp16 product
/// set is built; `out16` builds the pair that writes a 16-bit C. Refused by
/// code, nothing built, on a device without that element type's 16x16x16
/// cooperative matrices (bf16; or fp16 with `shaderFloat16`) or subgroup size
/// control, and for an fp16 C on a device that cannot declare the float
/// controls its epilogue's conversion needs.
fn nf4_matmul_kernels(state: &mut Inner, elem: Nf4Elem, out16: bool) -> Result<(), i32> {
    let slot = nf4_matmul_slot(&state.ctx, elem);
    let built_already = if out16 {
        state.nf4_matmul_out16[slot].is_some()
    } else {
        state.nf4_matmul[slot].is_some()
    };
    if built_already {
        return Ok(());
    }
    let report = &state.ctx.report;
    let capable = state.ctx.subgroup_size_control
        && match elem {
            Nf4Elem::Bf16 => report.bf16_cooperative_matrix,
            Nf4Elem::F16 => report.shader_float16 && report.f16_cooperative_matrix,
        };
    if !capable {
        set_error(format!(
            "nf4-matmul: this device has no {elem:?} cooperative-matrix product (or no subgroup size control)"
        ));
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    // an fp16 C is rounded by the device's own conversion under declared float
    // controls (both fp16 slots' epilogues); a device that cannot honour them
    // does not build them
    if out16 && elem == Nf4Elem::F16 && !state.ctx.fp16_rounding_declarable {
        set_error(
            "nf4-matmul-out16-float-controls: the fp16-output NF4 product declares RoundingModeRTE and DenormPreserve for 16- and 64-bit floats, and this device does not report all four; ack_matmul_nf4 and a conversion are",
        );
        return Err(ACK_ERR_UNSUPPORTED_OP);
    }
    let wide = state.ctx.can_require_subgroup_size(64) && !narrow_lanes_forced();
    let (set, width) = if wide {
        (
            if out16 {
                &SPV_NF4_MATMUL_OUT16_S64
            } else {
                &SPV_NF4_MATMUL_S64
            },
            64,
        )
    } else if state.ctx.can_require_subgroup_size(32) {
        (
            if out16 {
                &SPV_NF4_MATMUL_OUT16_S32
            } else {
                &SPV_NF4_MATMUL_S32
            },
            32,
        )
    } else {
        set_error("nf4-matmul: the device can require neither 64 nor 32 lanes");
        return Err(ACK_ERR_UNSUPPORTED_OP);
    };
    let mut built: Vec<Kernel> = Vec::with_capacity(2);
    for spv in set[slot].iter() {
        match state.ctx.kernel_with_subgroup(
            spv,
            9,
            crate::nf4_ops::MATMUL_PUSH_BYTES as u32,
            Some(width),
        ) {
            Ok(k) => built.push(k),
            Err(e) => {
                for k in built.drain(..) {
                    state.ctx.destroy_kernel(k);
                }
                set_error(format!(
                    "nf4-matmul pipeline ({elem:?}, {width} lanes): {e}"
                ));
                return Err(code_for(&e));
            }
        }
    }
    if out16 {
        state.nf4_matmul_out16[slot] = Some(built);
    } else {
        state.nf4_matmul[slot] = Some(built);
    }
    Ok(())
}

/// The kit's own cooperative-matrix product with its B operand held in
/// bitsandbytes' NF4 with double quantisation, read in its stored form
/// (nf4-matmul/v1, 2026-09-30): `groups` products in one dispatch, the z-th
/// writing `C[r, :] = A[r, :] . op(W_z)` for its rows r, in f32.
///
/// A holds `rows` rows of `k` 16-bit elements (bf16, `elem` 0, or fp16, 1) at a
/// pitch of `lda` elements (0 for `k`), from its buffer's front; C holds `rows`
/// rows of `n` f32. The stored form is `ack_nf4_decode`'s: `packed`,
/// `absmax_q`, `absmax2`, `tables`, `starts` (`members` int64) and `offsets`
/// (`members` f32), the source `src_base` values into the three stored
/// buffers. `groups_table` holds three u32 a group: its first row, its row
/// count (at most `max_rows`) and its weight's first value in the source (a
/// multiple of 8). The weight is `n x k` values stored row-major (`b_t` 1: C =
/// A . W^T) or `k x n` (`b_t` 0: C = A . W), each value decoded exactly as
/// `ack_nf4_decode` decodes it to `elem`. The kernel is the edge product's
/// with only B's staging changed, so each group's C is, bit for bit, that
/// decode followed by the kit's own product of the same operands.
///
/// The table is device data this entry does not read: the caller checks it (the
/// torch adapter does, on the host, before uploading it). The kernel still
/// guards it: a group whose rows run past `rows` or number more than
/// `max_rows`, whose weight runs past the source, or whose weight base is not a
/// multiple of 8, reads and writes nothing.
///
/// Refused by name before any device lookup: an unknown element type. After
/// it, the plan's names (`nf4-matmul-*`, `nf4-members-out-of-range`,
/// `nf4-base-out-of-range`, `nf4-source-too-small`), a buffer smaller than its
/// role (`nf4-matmul-<role>-too-small`), and an output that is one of the
/// inputs (`nf4-matmul-output-aliases-input`). A device without the element
/// type's cooperative matrices returns `ACK_ERR_UNSUPPORTED_OP`.
///
/// Since 2026-09-30 the staging caches each slot's member (a slot's value
/// index only grows, so the member is the cached one or a later one) and
/// unrolls a whole group of eight, and for fp16 a device that reports
/// `shaderFloat16` and can declare RoundingModeRTE and DenormPreserve for 16-
/// and 64-bit floats takes modules that round each decoded weight with its own
/// conversion (NF4_HW). The words are the same wherever no NaN is involved, and
/// a NaN is a NaN. `ACK_NF4_MATMUL_F16_HW=0` keeps the integer rounding.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move. The dispatch is RECORDED, so `ms_out` is NaN.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_nf4(
    dev: *const AckDevice,
    a: *const Buffer,
    packed: *const Buffer,
    absmax_q: *const Buffer,
    absmax2: *const Buffer,
    tables: *const Buffer,
    starts: *const Buffer,
    offsets: *const Buffer,
    groups_table: *const Buffer,
    c: *const Buffer,
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    b_t: u32,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    elem: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe {
        matmul_nf4_entry(
            false,
            dev,
            a,
            packed,
            absmax_q,
            absmax2,
            tables,
            starts,
            offsets,
            groups_table,
            c,
            rows,
            max_rows,
            n,
            k,
            b_t,
            lda,
            groups,
            members,
            src_base,
            elem,
            ms_out,
        )
    }
}

/// `ack_matmul_nf4` with C in the operands' own 16-bit type (2026-10-01): the
/// same call, arguments and rules, with C holding `rows` rows of `n` bf16 or
/// fp16 (`elem`) values written by the product's own epilogue -- bf16 rounded to
/// nearest even with a NaN kept quiet as cast_f32_bf16 rounds, fp16 by the
/// device's own conversion under declared float controls (to nearest even,
/// overflow to infinity, subnormals kept; a NaN stays a NaN, its sign and
/// payload free). So each group's C is `ack_matmul_nf4`'s f32 C rounded once:
/// nf4-decode/v1 followed by the kit's own 16-bit-output product, bit for bit,
/// without the f32 block and the conversion dispatch.
///
/// `n` must be even (`nf4-matmul-out16-odd-n`, refused by name before any
/// device lookup): C is written in pairs, and a pair must not straddle two
/// rows. Every other refusal is `ack_matmul_nf4`'s, with C counted in 2-byte
/// elements (`nf4-matmul-output-too-small`). An fp16 C needs the float controls
/// the fp16-output products need: a device that cannot declare
/// RoundingModeRTE and DenormPreserve for 16- and 64-bit floats returns
/// `ACK_ERR_UNSUPPORTED_OP` (`nf4-matmul-out16-float-controls`), and a caller
/// converts after `ack_matmul_nf4` instead. The staging follows
/// `ack_matmul_nf4`'s rule: the device's own conversion of each decoded fp16
/// weight where it can declare the float controls, and convert/v1's integer
/// routine under `ACK_NF4_MATMUL_F16_HW=0`.
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move. The dispatch is RECORDED, so `ms_out` is NaN.
/// # Safety
/// As `ack_matmul_nf4`'s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_matmul_nf4_to_16(
    dev: *const AckDevice,
    a: *const Buffer,
    packed: *const Buffer,
    absmax_q: *const Buffer,
    absmax2: *const Buffer,
    tables: *const Buffer,
    starts: *const Buffer,
    offsets: *const Buffer,
    groups_table: *const Buffer,
    c: *const Buffer,
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    b_t: u32,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    elem: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe {
        matmul_nf4_entry(
            true,
            dev,
            a,
            packed,
            absmax_q,
            absmax2,
            tables,
            starts,
            offsets,
            groups_table,
            c,
            rows,
            max_rows,
            n,
            k,
            b_t,
            lda,
            groups,
            members,
            src_base,
            elem,
            ms_out,
        )
    }
}

/// The body `ack_matmul_nf4` and `ack_matmul_nf4_to_16` share: `out16` writes C
/// in the operands' 16-bit type through the modules that round in their
/// epilogue, everything else the same.
/// # Safety
/// As `ack_matmul_nf4`'s.
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_nf4_entry(
    out16: bool,
    dev: *const AckDevice,
    a: *const Buffer,
    packed: *const Buffer,
    absmax_q: *const Buffer,
    absmax2: *const Buffer,
    tables: *const Buffer,
    starts: *const Buffer,
    offsets: *const Buffer,
    groups_table: *const Buffer,
    c: *const Buffer,
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    b_t: u32,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    elem: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let elem = match Nf4Elem::try_from(elem) {
        Ok(e) => e,
        Err(e) => {
            set_error(format!("nf4-matmul: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    if out16 && n % 2 != 0 {
        set_error(format!(
            "nf4-matmul: n = {n}; a 16-bit C is written in pairs, which an odd row would straddle: plan refused: nf4-matmul-out16-odd-n"
        ));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // the pair is built before the buffers are resolved: they borrow the guard
    if let Err(code) = nf4_matmul_kernels(&mut guard, elem, out16) {
        return code;
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let mut bound = Vec::with_capacity(10);
    for handle in [
        a,
        packed,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
        c,
    ] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let (a, packed, absmax_q, absmax2, tables, starts, offsets, groups_table, c) = (
        bound[0], bound[1], bound[2], bound[3], bound[4], bound[5], bound[6], bound[7], bound[8],
    );
    let covered = crate::nf4_ops::values_covered(packed.bytes, absmax_q.bytes, absmax2.bytes);
    let plan = match Nf4MatmulPlan::new(
        rows,
        max_rows,
        n,
        k,
        b_t != 0,
        lda,
        groups,
        members,
        src_base,
        covered,
        elem,
    ) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "nf4-matmul: {groups} group(s), {rows} rows (largest {max_rows}) x {n} x {k}, base {src_base}, \
                 {covered} covered: plan refused: {e}"
            ));
            return match e {
                Nf4MatmulError::SourceTooSmall | Nf4MatmulError::AddressLimitExceeded => {
                    ACK_ERR_SIZE
                }
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    // C's elements are f32, or 16-bit for the 16-bit C (n even, so whole words)
    let c_bytes = plan.c_elements() * if out16 { 2 } else { 4 };
    for (buffer, needed, name) in [
        (a, 2 * plan.a_elements(), "a"),
        (tables, 4 * TABLE_ENTRIES as u64, "tables"),
        (starts, 8 * u64::from(members), "starts"),
        (offsets, 4 * u64::from(members), "offsets"),
        (
            groups_table,
            4 * u64::from(GROUP_WORDS) * u64::from(groups),
            "groups",
        ),
        (c, c_bytes, "output"),
    ] {
        if buffer.bytes < needed {
            set_error(format!(
                "nf4-matmul: {name} of {} bytes for {needed}: plan refused: nf4-matmul-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the kernel writes C while other workgroups still read the inputs
    let inputs = [
        a,
        packed,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
    ];
    if inputs.iter().any(|b| b.id() == c.id()) {
        set_error("nf4-matmul: plan refused: nf4-matmul-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let set = if out16 {
        &guard.nf4_matmul_out16
    } else {
        &guard.nf4_matmul
    };
    let kernel = match set[nf4_matmul_slot(&guard.ctx, elem)].as_ref() {
        Some(pair) => &pair[plan.b_t() as usize],
        None => {
            set_error("nf4-matmul: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    // bindings 0-8: A, the packed indices, C, then the rest of the stored form
    // and the groups table (the shader's NF4_B note)
    let slots = [
        a,
        packed,
        c,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
    ];
    if let Err(e) = guard.ctx.bind(kernel, &slots) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(a.id()),
        Access::read(packed.id()),
        Access::write(c.id()),
        Access::read(absmax_q.id()),
        Access::read(absmax2.id()),
        Access::read(tables.id()),
        Access::read(starts.id()),
        Access::read(offsets.id()),
        Access::read(groups_table.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// The NF4 product at one to 16 rows a group, as decoding runs it
/// (nf4-gemv/v1, 2026-09-30): `groups` products in one dispatch, the z-th
/// writing `C[r, :] = A[r, :] . W_z^T` for its rows r, in f32, with W_z stored
/// `n x k`.
///
/// The operands, the stored form and the groups table are `ack_matmul_nf4`'s
/// (its forward form), with A's pitch a multiple of 8 and every group at most
/// 16 rows. One workgroup of 64 invocations computes one column of one group.
/// THE LAW (the kernel's header): each weight decoded exactly as
/// `ack_nf4_decode` decodes it to `elem` and widened, each product exact in
/// f32, each invocation's words summed in ascending order, then a fixed
/// pairwise tree over the 64 partial sums. A host that follows that order gets
/// the same words; it is not the cooperative-matrix product's order, and
/// against `ack_matmul_nf4` the two agree to f32 summation error. The kernel
/// needs no cooperative matrices, no 16-bit arithmetic and no subgroup size.
/// For fp16 a device that reports `shaderFloat16` and can declare
/// RoundingModeRTE and DenormPreserve for 16- and 64-bit floats takes the
/// modules that convert with its own instructions (F16_HW, 2026-09-30; the
/// integer modules ran 2.8-3.5x slower than bf16's at the same shapes on an RX
/// 9070 XT); every word that is not a NaN is the same, and a NaN is a NaN.
/// `ACK_NF4_GEMV_F16_HW=0` keeps the integer modules.
///
/// The kernel guards the table as `ack_matmul_nf4` does (rows, source, a
/// weight base off a multiple of 8), and writes nothing for a group of no rows
/// or more than 16. Refusals: an unknown element type before
/// any device lookup; after it, nf4-matmul/v1's names for the forward form,
/// `nf4-gemv-rows-above-16`, `nf4-gemv-pitch-not-whole-words`, a buffer
/// smaller than its role (`nf4-matmul-<role>-too-small`) and an output that is
/// one of the inputs (`nf4-matmul-output-aliases-input`).
///
/// OPTIONAL, resolved by name and in no pointer table: `ACK_ABI_VERSION` does
/// not move. The dispatch is RECORDED, so `ms_out` is NaN.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_gemv_nf4(
    dev: *const AckDevice,
    a: *const Buffer,
    packed: *const Buffer,
    absmax_q: *const Buffer,
    absmax2: *const Buffer,
    tables: *const Buffer,
    starts: *const Buffer,
    offsets: *const Buffer,
    groups_table: *const Buffer,
    c: *const Buffer,
    rows: u32,
    max_rows: u32,
    n: u32,
    k: u32,
    lda: u32,
    groups: u32,
    members: u32,
    src_base: u32,
    elem: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let elem = match Nf4Elem::try_from(elem) {
        Ok(e) => e,
        Err(e) => {
            set_error(format!("nf4-gemv: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let slot = gemv_slot(&guard.ctx, elem);
    // the module for the call's largest group, built before the buffers are
    // resolved (they borrow the guard); a count no module holds is refused by
    // the plan below, by name, before any kernel is used
    let module = crate::nf4_ops::gemv_module(max_rows).unwrap_or(2);
    if guard.nf4_gemv[slot][module].is_none() {
        match guard.ctx.kernel(
            SPV_NF4_GEMV[slot][module],
            9,
            crate::nf4_ops::GEMV_PUSH_BYTES as u32,
        ) {
            Ok(k) => guard.nf4_gemv[slot][module] = Some(k),
            Err(e) => {
                set_error(format!(
                    "nf4-gemv pipeline ({elem:?}, module {module}): {e}"
                ));
                return code_for(&e);
            }
        }
    }
    let mut bound = Vec::with_capacity(9);
    for handle in [
        a,
        packed,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
        c,
    ] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let (a, packed, absmax_q, absmax2, tables, starts, offsets, groups_table, c) = (
        bound[0], bound[1], bound[2], bound[3], bound[4], bound[5], bound[6], bound[7], bound[8],
    );
    let covered = crate::nf4_ops::values_covered(packed.bytes, absmax_q.bytes, absmax2.bytes);
    let plan = match Nf4GemvPlan::new(
        rows, max_rows, n, k, lda, groups, members, src_base, covered, elem,
    ) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "nf4-gemv: {groups} group(s), {rows} rows (largest {max_rows}) x {n} x {k}, base {src_base}, \
                 {covered} covered: plan refused: {e}"
            ));
            return match e {
                Nf4MatmulError::SourceTooSmall | Nf4MatmulError::AddressLimitExceeded => {
                    ACK_ERR_SIZE
                }
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    for (buffer, needed, name) in [
        (a, 2 * plan.a_elements(), "a"),
        (tables, 4 * TABLE_ENTRIES as u64, "tables"),
        (starts, 8 * u64::from(members), "starts"),
        (offsets, 4 * u64::from(members), "offsets"),
        (
            groups_table,
            4 * u64::from(GROUP_WORDS) * u64::from(groups),
            "groups",
        ),
        (c, 4 * plan.c_elements(), "output"),
    ] {
        if buffer.bytes < needed {
            set_error(format!(
                "nf4-gemv: {name} of {} bytes for {needed}: plan refused: nf4-matmul-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    let inputs = [
        a,
        packed,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
    ];
    if inputs.iter().any(|b| b.id() == c.id()) {
        set_error("nf4-gemv: plan refused: nf4-matmul-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.nf4_gemv[slot][plan.module()].as_ref() {
        Some(k) => k,
        None => {
            set_error("nf4-gemv: kernel absent on this device");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    let slots = [
        a,
        packed,
        c,
        absmax_q,
        absmax2,
        tables,
        starts,
        offsets,
        groups_table,
    ];
    if let Err(e) = guard.ctx.bind(kernel, &slots) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::read(a.id()),
        Access::read(packed.id()),
        Access::write(c.id()),
        Access::read(absmax_q.id()),
        Access::read(absmax2.id()),
        Access::read(tables.id()),
        Access::read(starts.id()),
        Access::read(offsets.id()),
        Access::read(groups_table.id()),
    ];
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// expert-factor-f16/v1 (2026-09-29): the sub-bit student's expert factors, each
/// built from its signs in one pass (ops 0, 1), and their backward (ops 2-4).
///
/// For each routed expert `q < p` (its id in `ids`, int64, below `experts`) and
/// factor row `m`, the forward writes columns `[off, off + r8)` of the fp16
/// factor (`p` x `m` x `ld`): `sign * f16(row * col)` for a column below `r`,
/// 0 past it. The sign is the fp32 latent's (op 0, `W >= 0`) or the packed
/// cache's bit (op 1). An absent scale is 1 (`flags` bit 0 a row scale, bit 1 a
/// column scale, which needs a row scale). With `flags` bit 2 (2026-09-30) the
/// cache ops 1, 3 and 4 read an int8 cache (`experts` x `m` x `r` bytes), one
/// signed byte a sign, +1 or -1; any other byte makes its entry and its sums
/// NaN. The backward reads the fp16 upstream
/// gradient at the same columns:
/// - op 2 writes the latent's gradient (the SmoothSign surrogate at slope `k`)
///   into `out_a` (`experts` x `m` x `r` f32, rows of experts not routed
///   untouched), and into `out_b` the row scale's gradient (a row scale only)
///   or `g * sign` whole (both scales), for the caller's reductions;
/// - op 3 writes the row scale's gradient into `out_a`;
/// - op 4 writes the column scale's gradient into `out_a`.
///
/// Every buffer is a live handle. A role an operation does not use may be any
/// buffer, but an output role must not be one of the inputs.
///
/// Refused by name, before any device lookup, through `expert_factor_ops`: a
/// zero dimension, an operation or flags out of range, a rank past the
/// family's bound, a padded rank that is not the rank rounded up to 8, an odd
/// or overflowing layout, a column scale without a row scale, int8 signs with a
/// latent operation (`expert-factor-int8-signs-with-latent`), and the element
/// and grid bounds. After the lookup: a buffer smaller than its role
/// (`expert-factor-<role>-too-small`), an output that is one of the inputs
/// (`expert-factor-output-aliases-input`), and a pipeline this device cannot
/// build (`expert-factor-pipeline-unavailable`, ACK_ERR_UNSUPPORTED_OP).
///
/// OPTIONAL, resolved by name: `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_expert_factor_f16(
    dev: *const AckDevice,
    src: *const Buffer,
    ids: *const Buffer,
    row: *const Buffer,
    col: *const Buffer,
    factor: *const Buffer,
    out_a: *const Buffer,
    out_b: *const Buffer,
    op: u32,
    flags: u32,
    experts: u32,
    p: u32,
    m: u32,
    r: u32,
    r8: u32,
    ld: u32,
    off: u32,
    k: f32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let op = match ExpertFactorOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("expert_factor: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match ExpertFactorPlan::new(op, flags, experts, p, m, r, r8, ld, off, k) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "expert_factor: op {op:?}, {p} of {experts} experts x {m} rows, rank {r} ({r8}) at {off} of {ld}: plan refused: {e}"
            ));
            return match e {
                ExpertFactorPlanError::ElementLimitExceeded
                | ExpertFactorPlanError::GridAboveLimit => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.expert_factor.is_none() {
        match guard.ctx.kernel(
            SPV_EXPERT_FACTOR,
            7,
            crate::expert_factor_ops::PUSH_BYTES as u32,
        ) {
            Ok(k) => guard.expert_factor = Some(k),
            Err(e) => {
                set_error(format!(
                    "expert_factor pipeline ({e}): plan refused: expert-factor-pipeline-unavailable"
                ));
                return ACK_ERR_UNSUPPORTED_OP;
            }
        }
    }
    let mut bound = Vec::with_capacity(7);
    for handle in [src, ids, row, col, factor, out_a, out_b] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let (src, ids, row, col) = (bound[0], bound[1], bound[2], bound[3]);
    let (factor, out_a, out_b) = (bound[4], bound[5], bound[6]);
    for (buffer, needed_bytes, name) in [
        (src, plan.src_bytes(), "source"),
        (ids, plan.p() as u64 * 8, "ids"),
        (row, plan.row_bytes(), "row"),
        (col, plan.col_bytes(), "col"),
        (factor, plan.factor_bytes(), "factor"),
        (out_a, plan.out_a_bytes(), "out-a"),
        (out_b, plan.out_b_bytes(), "out-b"),
    ] {
        if buffer.bytes < needed_bytes {
            set_error(format!(
                "expert_factor: {name} of {} bytes for {needed_bytes}: plan refused: expert-factor-{name}-too-small",
                buffer.bytes
            ));
            return ACK_ERR_SIZE;
        }
    }
    // the forward writes the factor; the backward writes out_a and out_b. None
    // of what an operation writes may be what it reads, nor one another.
    let forward = matches!(
        op,
        ExpertFactorOp::LatentForward | ExpertFactorOp::PackedForward
    );
    let reads = if forward {
        vec![src.id(), ids.id(), row.id(), col.id()]
    } else {
        vec![src.id(), ids.id(), row.id(), col.id(), factor.id()]
    };
    let writes = if forward {
        vec![factor.id()]
    } else {
        vec![out_a.id(), out_b.id()]
    };
    if writes.iter().any(|w| reads.contains(w)) || (!forward && out_a.id() == out_b.id()) {
        set_error("expert_factor: plan refused: expert-factor-output-aliases-input");
        return ACK_ERR_SHAPE;
    }
    let kernel = match guard.expert_factor.as_ref() {
        Some(k) => k,
        None => {
            set_error("expert_factor: plan refused: expert-factor-pipeline-unavailable");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    if let Err(e) = guard
        .ctx
        .bind(kernel, &[src, ids, row, col, factor, out_a, out_b])
    {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = if forward {
        [
            Access::read(src.id()),
            Access::read(ids.id()),
            Access::read(row.id()),
            Access::read(col.id()),
            Access::write(factor.id()),
            Access::read(out_a.id()),
            Access::read(out_b.id()),
        ]
    } else {
        [
            Access::read(src.id()),
            Access::read(ids.id()),
            Access::read(row.id()),
            Access::read(col.id()),
            Access::read(factor.id()),
            Access::write(out_a.id()),
            Access::write(out_b.id()),
        ]
    };
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

/// student-norm-f16/v1 (2026-09-29): the sub-bit student's attention-block
/// RMSNorms, the residual add before the post-attention one, and its q/k norms
/// with RoPE, forward and backward (`kernels/student_norm_f16.comp` states each
/// operation's law; research/subbit_expert/moe/student.py's O12 Fusables are the
/// reference).
///
/// Eight buffer roles: x, aux, w, tab, out0, out1, dy, io2. A role an operation
/// does not use may be any live buffer. What an operation writes (out0, out1, and
/// io2 for op 0 with ADD) must be none of what it reads, nor one another.
///
/// Refused by name before any device lookup (`student-norm-<reason>`: dimension,
/// odd width, width above the limit, operation, flags, positions, rows per
/// group, epsilon, element and grid bounds); after it, a buffer smaller than its
/// role (`student-norm-<role>-too-small`), an output aliasing an input or another
/// output (`student-norm-output-aliases-input`), and a pipeline this device
/// cannot build (`student-norm-pipeline-unavailable`, ACK_ERR_UNSUPPORTED_OP).
///
/// OPTIONAL, resolved by name: `ACK_ABI_VERSION` does not move.
/// # Safety
/// `dev` and the buffers are handles from this library, or null; `ms_out` is
/// null or points to one writable f64.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_student_norm_f16(
    dev: *const AckDevice,
    x: *const Buffer,
    aux: *const Buffer,
    w: *const Buffer,
    tab: *const Buffer,
    out0: *const Buffer,
    out1: *const Buffer,
    dy: *const Buffer,
    io2: *const Buffer,
    op: u32,
    flags: u32,
    rows: u32,
    d: u32,
    eps: f32,
    t: u32,
    h: u32,
    rows_per_group: u32,
    ms_out: *mut f64,
) -> i32 {
    unsafe { no_duration(ms_out) };
    let op = match NormOp::try_from(op) {
        Ok(op) => op,
        Err(e) => {
            set_error(format!("student_norm: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let plan = match NormPlan::new(op, flags, rows, d, eps, t, h, rows_per_group) {
        Ok(plan) => plan,
        Err(e) => {
            set_error(format!(
                "student_norm: op {op:?}, {rows} rows of {d}: plan refused: {e}"
            ));
            return match e {
                NormPlanError::ElementLimitExceeded | NormPlanError::GridAboveLimit => ACK_ERR_SIZE,
                _ => ACK_ERR_SHAPE,
            };
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.student_norm.is_none() {
        match guard.ctx.kernel(
            SPV_STUDENT_NORM,
            8,
            crate::student_norm_ops::PUSH_BYTES as u32,
        ) {
            Ok(k) => guard.student_norm = Some(k),
            Err(e) => {
                set_error(format!(
                    "student_norm pipeline ({e}): plan refused: student-norm-pipeline-unavailable"
                ));
                return ACK_ERR_UNSUPPORTED_OP;
            }
        }
    }
    let mut bound = Vec::with_capacity(8);
    for handle in [x, aux, w, tab, out0, out1, dy, io2] {
        match buffer_of(&device, &guard, handle) {
            Ok(b) => bound.push(b),
            Err(code) => return code,
        }
    }
    let names = ["x", "aux", "w", "tab", "out0", "out1", "dy", "io2"];
    let need = plan.role_bytes();
    for i in 0..8 {
        if bound[i].bytes < need[i] {
            set_error(format!(
                "student_norm: {} of {} bytes for {}: plan refused: student-norm-{}-too-small",
                names[i], bound[i].bytes, need[i], names[i]
            ));
            return ACK_ERR_SIZE;
        }
    }
    let writes = plan.writes();
    for i in 0..8 {
        if !writes[i] {
            continue;
        }
        for j in 0..8 {
            if j != i && bound[j].id() == bound[i].id() && (need[j] > 0 || writes[j]) {
                set_error("student_norm: plan refused: student-norm-output-aliases-input");
                return ACK_ERR_SHAPE;
            }
        }
    }
    let kernel = match guard.student_norm.as_ref() {
        Some(k) => k,
        None => {
            set_error("student_norm: plan refused: student-norm-pipeline-unavailable");
            return ACK_ERR_UNSUPPORTED_OP;
        }
    };
    let buffers: [&Buffer; 8] = [
        bound[0], bound[1], bound[2], bound[3], bound[4], bound[5], bound[6], bound[7],
    ];
    if let Err(e) = guard.ctx.bind(kernel, &buffers) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access: Vec<Access> = (0..8)
        .map(|i| {
            if writes[i] {
                Access::write(buffers[i].id())
            } else {
                Access::read(buffers[i].id())
            }
        })
        .collect();
    recorded(guard.ctx.dispatch(
        kernel,
        &plan.push_constants(),
        plan.dispatch_groups(),
        1,
        &access,
    ))
}

#[allow(clippy::too_many_arguments)]
unsafe fn loss_entry(
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
    flags: u32,
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
    // bf16 storage (ack_loss_bf16, 2026-09-28): refused by name before any lookup
    let primary_bf16 = flags & ACK_LOSS_PRIMARY_BF16 != 0;
    let output_bf16 = flags & ACK_LOSS_OUTPUT_BF16 != 0;
    let refusal = if flags & !(ACK_LOSS_PRIMARY_BF16 | ACK_LOSS_OUTPUT_BF16) != 0 {
        Some("loss-bf16-unknown-flag")
    } else if primary_bf16 && !matches!(op, LossOp::Forward | LossOp::Backward) {
        Some("loss-bf16-logits-op")
    } else if output_bf16 && (op != LossOp::Backward || !primary_bf16) {
        Some("loss-bf16-output-op")
    } else if output_bf16 && cols % 2 != 0 {
        Some("loss-bf16-output-odd-cols")
    } else {
        None
    };
    if let Some(name) = refusal {
        set_error(format!("loss: flags {flags:#x}: plan refused: {name}"));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if flags != 0 {
        if let Err(code) = loss_bf16_kernels(&mut guard) {
            return code;
        }
    }
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
    // `width` is the element's bytes: 2 for a bf16 buffer, 4 otherwise
    let capacity = |buffer: &Buffer, needed: usize, name: &str, width: u64| -> Result<(), i32> {
        if buffer.bytes / width < needed as u64 {
            set_error(format!(
                "loss: {name} of {} elements for {needed}: plan refused: loss-{name}-too-small",
                buffer.bytes / width
            ));
            return Err(ACK_ERR_SIZE);
        }
        Ok(())
    };
    if reads_primary {
        let width = if primary_bf16 { 2 } else { 4 };
        if let Err(code) = capacity(primary, plan.primary_len(), "primary", width) {
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
        if let Err(code) = capacity(upstream, plan.upstream_len(), "upstream", 4) {
            return code;
        }
    }
    let width = if output_bf16 { 2 } else { 4 };
    if let Err(code) = capacity(out, plan.output_len(), "output", width) {
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
    let kernel = match if flags == 0 {
        guard.loss.as_ref()
    } else {
        guard
            .loss_bf16
            .as_ref()
            .map(|set| &set[usize::from(output_bf16)])
    } {
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

/// GradScaler's non-finite check and unscale of ONE dense float32 gradient
/// (amp-unscale-f32/v1, 2026-09-29): every element that is an infinity or a
/// NaN stores 1.0f into `found`'s element, and every element is multiplied in
/// place by `scale`'s element unless that is exactly 1.0f, which keeps every
/// bit. The product is binary32 in integer instructions with torch's CPU NaN
/// rules (`kernels/amp_unscale_f32.comp`), so its bits are the CPU kernel's.
/// `request` is an `AckAmpUnscaleRequest` of `amp_ops::REQUEST_BYTES` bytes:
/// the gradient's element offset and count, then the scale's and `found`'s
/// element offsets. Refused by name before any dispatch: a block of another
/// size (`amp-request-size`), a word above the kernel's limits
/// (`amp-count-above-limit`, `amp-word-above-u32`), a view past a buffer
/// (`amp-data-too-small`, `amp-scale-too-small`, `amp-found-too-small`), and
/// two slots sharing a buffer (`amp-buffers-alias`: the kernel writes the
/// gradient and `found` while other workgroups still read). An empty gradient
/// dispatches nothing.
///
/// OPTIONAL, resolved by name and in no pointer table, like `ack_loss_bf16`:
/// `ACK_ABI_VERSION` does not move. The dispatch is RECORDED, so `ms_out` is
/// NaN.
/// # Safety
/// `request` must point to `request_size` readable bytes; `ms_out` must be
/// null or point to a writable f64.
#[no_mangle]
pub unsafe extern "C" fn ack_amp_unscale(
    dev: *const AckDevice,
    data: *const Buffer,
    scale: *const Buffer,
    found: *const Buffer,
    request: *const u8,
    request_size: usize,
    ms_out: *mut f64,
) -> i32 {
    if request.is_null() {
        return ACK_ERR_NULL;
    }
    // the request is decoded before any device lookup, as every family here does
    let block = unsafe { std::slice::from_raw_parts(request, request_size) };
    let Some(r) = AmpUnscaleRequest::from_bytes(block) else {
        set_error(format!(
            "amp unscale: a request block of {request_size} bytes, not {}: plan refused: amp-request-size",
            amp_ops::REQUEST_BYTES
        ));
        return ACK_ERR_SHAPE;
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.amp_unscale.is_none() {
        match guard
            .ctx
            .kernel(SPV_AMP_UNSCALE, 3, amp_ops::PUSH_BYTES as u32)
        {
            Ok(k) => guard.amp_unscale = Some(k),
            Err(e) => {
                set_error(format!("amp unscale pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let data = match buffer_of(&device, &guard, data) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let scale = match buffer_of(&device, &guard, scale) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let found = match buffer_of(&device, &guard, found) {
        Ok(b) => b,
        Err(code) => return code,
    };
    if data.id() == scale.id() || data.id() == found.id() || scale.id() == found.id() {
        set_error("amp unscale: plan refused: amp-buffers-alias");
        return ACK_ERR_SHAPE;
    }
    let plan = match AmpPlan::new(&r, data.bytes / 4, scale.bytes / 4, found.bytes / 4) {
        Ok(p) => p,
        Err(e) => {
            // the detail precedes the name, so a consumer reading the last
            // colon-separated word (the torch adapter's refusal histogram) gets it
            set_error(format!(
                "amp unscale: {r:?} over {} / {} / {} elements: plan refused: {}",
                data.bytes / 4,
                scale.bytes / 4,
                found.bytes / 4,
                e.name()
            ));
            return ACK_ERR_SIZE;
        }
    };
    if plan.is_empty() {
        unsafe { no_duration(ms_out) };
        return ACK_OK;
    }
    let Some(kernel) = guard.amp_unscale.as_ref() else {
        set_error("amp unscale: kernel absent on this device");
        return ACK_ERR_UNSUPPORTED_OP;
    };
    if let Err(e) = guard.ctx.bind(kernel, &[data, scale, found]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::write(data.id()),
        Access::read(scale.id()),
        Access::write(found.id()),
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

/// GradScaler's scale update (amp-update-scale-f32/v1, 2026-09-29), torch's
/// `_amp_update_scale_` in one invocation over three one-element operands: when
/// `found`'s float32 is nonzero (a NaN included, -0.0 not) the scale is
/// multiplied by the backoff factor and the int32 growth tracker is set to 0;
/// otherwise the tracker is incremented (wrapping) and, when that equals the
/// growth interval, the scale is multiplied by the growth factor if the product
/// is finite and the tracker is set to 0. The products are binary32 in integer
/// instructions with torch's CPU NaN rules (`kernels/amp_update_scale_f32.comp`),
/// so their bits are the CPU kernel's for every factor this entry admits.
/// `request` is an `AckAmpUpdateRequest` of `amp_ops::UPDATE_REQUEST_BYTES`
/// bytes: the scale's, the tracker's and `found`'s element offsets, then the two
/// factors as doubles and the interval as an int64, as torch passes them.
/// Refused by name before any dispatch: a block of another size
/// (`amp-request-size`), a factor that is not a float32 exactly
/// (`amp-update-factor-not-f32`: the CPU's double product would round twice), an
/// interval outside int32 (`amp-update-interval-above-i32`), an offset past the
/// kernel's words or its buffer (`amp-word-above-u32`, `amp-scale-too-small`,
/// `amp-tracker-too-small`, `amp-found-too-small`), and two slots sharing a
/// buffer (`amp-buffers-alias`).
///
/// OPTIONAL, resolved by name and in no pointer table, like `ack_amp_unscale`:
/// `ACK_ABI_VERSION` does not move. The dispatch is RECORDED, so `ms_out` is
/// NaN.
/// # Safety
/// `request` must point to `request_size` readable bytes; `ms_out` must be
/// null or point to a writable f64.
#[no_mangle]
pub unsafe extern "C" fn ack_amp_update_scale(
    dev: *const AckDevice,
    scale: *const Buffer,
    tracker: *const Buffer,
    found: *const Buffer,
    request: *const u8,
    request_size: usize,
    ms_out: *mut f64,
) -> i32 {
    if request.is_null() {
        return ACK_ERR_NULL;
    }
    // the request is decoded before any device lookup, as every family here does
    let block = unsafe { std::slice::from_raw_parts(request, request_size) };
    let Some(r) = AmpUpdateRequest::from_bytes(block) else {
        set_error(format!(
            "amp update: a request block of {request_size} bytes, not {}: plan refused: amp-request-size",
            amp_ops::UPDATE_REQUEST_BYTES
        ));
        return ACK_ERR_SHAPE;
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.amp_update.is_none() {
        match guard
            .ctx
            .kernel(SPV_AMP_UPDATE, 3, amp_ops::UPDATE_PUSH_BYTES as u32)
        {
            Ok(k) => guard.amp_update = Some(k),
            Err(e) => {
                set_error(format!("amp update pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let scale = match buffer_of(&device, &guard, scale) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let tracker = match buffer_of(&device, &guard, tracker) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let found = match buffer_of(&device, &guard, found) {
        Ok(b) => b,
        Err(code) => return code,
    };
    if scale.id() == tracker.id() || scale.id() == found.id() || tracker.id() == found.id() {
        set_error("amp update: plan refused: amp-buffers-alias");
        return ACK_ERR_SHAPE;
    }
    let plan = match AmpUpdatePlan::new(&r, scale.bytes / 4, tracker.bytes / 4, found.bytes / 4) {
        Ok(p) => p,
        Err(e) => {
            // the detail precedes the name, so a consumer reading the last
            // colon-separated word (the torch adapter's refusal histogram) gets it
            set_error(format!(
                "amp update: {r:?} over {} / {} / {} elements: plan refused: {}",
                scale.bytes / 4,
                tracker.bytes / 4,
                found.bytes / 4,
                e.name()
            ));
            return ACK_ERR_SIZE;
        }
    };
    let Some(kernel) = guard.amp_update.as_ref() else {
        set_error("amp update: kernel absent on this device");
        return ACK_ERR_UNSUPPORTED_OP;
    };
    if let Err(e) = guard.ctx.bind(kernel, &[scale, tracker, found]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let access = [
        Access::write(scale.id()),
        Access::write(tracker.id()),
        Access::read(found.id()),
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

fn index_plan_code(err: index_ops::IndexPlanError) -> i32 {
    match err {
        index_ops::IndexPlanError::AddressOverflow
        | index_ops::IndexPlanError::StatusSlotOutOfRange => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// One `ack_index` call's request (index/v1, ABI 12): what the caller asks
/// for, not a finished push block, so the kit's own planner (`index_ops`)
/// derives the grouping, the unit count and the dispatch, and no binding has
/// to restate that arithmetic. Shapes, offsets and strides are in ELEMENTS of
/// their own tensor. `op` is 0 gather, 1 scatter, 2 scatter-add; `dtype` the
/// storage code of `index_ops::Dtype`; `flags` the `index_ops::FLAG_*` bits;
/// `value` the scalar source in the storage's own bits when `FLAG_VALUE` is
/// set, and `alpha_bits` the f32 scale when `FLAG_ALPHA` is. 200 bytes, every
/// field naturally aligned, no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AckIndexRequest {
    pub op: u32,
    pub dtype: u32,
    pub ndim: u32,
    pub dim: u32,
    pub flags: u32,
    pub status_slot: u32,
    pub shape: [u64; 4],
    pub index_offset: u64,
    pub index_strides: [u64; 4],
    pub src_offset: u64,
    pub src_strides: [u64; 4],
    pub out_offset: u64,
    pub out_strides: [u64; 4],
    pub bound: u64,
    pub value: u64,
    pub alpha_bits: u32,
    /// Zero; refused otherwise.
    pub reserved: u32,
}

/// Gather, scatter or scatter-add over strided views of at most four
/// dimensions (index/v1 at the C ABI, schema 2; ABI 12): torch's gather,
/// index_select, scatter, index_copy_, scatter_add and index_add, for elements
/// of 1, 2, 4 or 8 bytes and int32 or int64 indices, as `index_ops` and the
/// shader's header describe them. `request` (`request_size` must be
/// `size_of::<AckIndexRequest>()`) is planned here against the four buffers'
/// real sizes and identities.
///
/// UNLIKE THE EMBEDDING AND LOSS FAMILIES, THE INDICES STAY ON THE DEVICE. The
/// kernel checks each one against the plan's bound and never addresses with
/// one out of range: a gather writes the all-ones pattern there and a scatter
/// skips the write, and either ORs `index_ops::STATUS_INDEX` into word
/// `status_slot` of `status`. The caller owns that buffer, zeroes a slot
/// before using it, and reads it back at its next wait, which is when it
/// learns of the error: torch's CUDA semantics (an asynchronous device
/// assert), not its CPU ones (an error before running). `status` may be no
/// other operand's buffer, and `out` neither `src`'s (unless the plan's scalar
/// source flag leaves `src` unread) nor `index`'s.
///
/// Refusals are by name before any dispatch: `ACK_ERR_NULL` for a null
/// request, `ACK_ERR_ABI` for a request of another size, `ACK_ERR_SIZE` for a
/// view past its buffer or the 2^31 word limit (`index-address-overflow`) or a
/// status slot past its buffer (`index-status-slot-out-of-range`), and
/// `ACK_ERR_SHAPE` for a nonzero reserved word and everything else
/// `index_ops::IndexPlanError` names; the operation, storage and flags are
/// decoded before any device lookup. The module is built on the first call,
/// so a failure to build it is this call's error. The dispatch is RECORDED and
/// runs at the next flush, so `ms_out` is NaN.
/// # Safety
/// `request` must point to `request_size` readable bytes; `ms_out` may be null.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ack_index(
    dev: *const AckDevice,
    src: *const Buffer,
    index: *const Buffer,
    out: *const Buffer,
    status: *const Buffer,
    request: *const AckIndexRequest,
    request_size: usize,
    ms_out: *mut f64,
) -> i32 {
    if request.is_null() {
        return ACK_ERR_NULL;
    }
    if request_size != std::mem::size_of::<AckIndexRequest>() {
        set_error(format!(
            "index: the request is {request_size} bytes, not this library's {}",
            std::mem::size_of::<AckIndexRequest>()
        ));
        return ACK_ERR_ABI;
    }
    // the caller promised `request_size` readable bytes, not alignment
    let r = unsafe { std::ptr::read_unaligned(request) };
    if r.reserved != 0 {
        set_error("index: plan refused: index-reserved-word-nonzero");
        return ACK_ERR_SHAPE;
    }
    // the words that need no buffer are decoded before any device lookup
    if let Err(e) = index_ops::IndexOp::try_from(r.op)
        .and_then(|_| index_ops::Dtype::try_from(r.dtype))
        .and_then(|_| index_ops::request_flags(r.flags))
    {
        set_error(format!("index: plan refused: {e}"));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    // mutable only to build the module on first use, which happens before any
    // buffer is borrowed from the guard
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if guard.index.is_none() {
        let state: &mut Inner = &mut guard;
        match state
            .ctx
            .kernel(SPV_INDEX, index_ops::BINDINGS, index_ops::PUSH_BYTES as u32)
        {
            Ok(k) => state.index = Some(k),
            Err(e) => {
                set_error(format!("index pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    // Schema 3: a grouped scatter-add sorts with the radix module, built here
    // before any buffer is borrowed whenever the words alone allow that plan.
    if guard.radix.is_none() && index_ops::may_plan_grouped(r.op, r.dtype, r.ndim, r.dim, r.shape) {
        let state: &mut Inner = &mut guard;
        match state
            .ctx
            .kernel(SPV_RADIX, radix_ops::BINDINGS, radix_ops::PUSH_BYTES as u32)
        {
            Ok(k) => state.radix = Some(k),
            Err(e) => {
                set_error(format!("radix pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    // resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle
    let src = match buffer_of(&device, &guard, src) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let index = match buffer_of(&device, &guard, index) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let status = match buffer_of(&device, &guard, status) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let storage = |b: &Buffer| index_ops::Storage::new(b.id(), b.bytes);
    let bindings = index_ops::Bindings {
        src: storage(src),
        index: storage(index),
        out: storage(out),
        status: storage(status),
    };
    let plan = match IndexPlan::from_abi(
        r.op,
        r.dtype,
        r.ndim,
        r.dim,
        r.flags,
        r.status_slot,
        r.shape,
        index_ops::View::new(r.index_offset, r.index_strides),
        index_ops::View::new(r.src_offset, r.src_strides),
        index_ops::View::new(r.out_offset, r.out_strides),
        r.bound,
        r.value,
        r.alpha_bits,
        bindings,
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("index: plan refused: {e}"));
            return index_plan_code(e);
        }
    };
    let kernel = guard.index.as_ref().expect("built above");
    let rc = if plan.strategy() == index_ops::IndexStrategy::Grouped {
        let Some(sorter) = guard.radix.as_ref() else {
            // may_plan_grouped and the planner disagree: a defect, named
            set_error("index: a grouped plan without the radix module (may_plan_grouped)");
            return ACK_ERR_UNSUPPORTED_OP;
        };
        // the radix module's scratch, the kit's own for this call and retired
        // into the batch at once
        let scratch = match guard.ctx.buffer(plan.scratch_words() * 4) {
            Ok(b) => b,
            Err(e) => {
                set_error(format!(
                    "index: a grouped scatter-add's scratch of {} words: {e}",
                    plan.scratch_words()
                ));
                return code_for(&e);
            }
        };
        let rc = record_grouped(
            &guard,
            [kernel, sorter],
            &plan,
            [src, index, out, status],
            &scratch,
        );
        guard.ctx.destroy_buffer(scratch);
        rc
    } else if plan.strategy() == index_ops::IndexStrategy::Elected {
        // Schema 3: the elected scatter's scratch, one word a destination, is
        // the kit's own for this call: retired into the batch at once, so the
        // pool takes it back after the batch that reads it completes.
        let scratch = match guard.ctx.buffer(plan.scratch_words() * 4) {
            Ok(b) => b,
            Err(e) => {
                set_error(format!(
                    "index: an elected scatter's scratch of {} words: {e}",
                    plan.scratch_words()
                ));
                return code_for(&e);
            }
        };
        let rc = record_elected(&guard, kernel, &plan, [src, index, out, status], &scratch);
        guard.ctx.destroy_buffer(scratch);
        rc
    } else {
        // one pass: the fiber walk, or an integer scatter-add's atomics
        if let Err(e) = guard.ctx.bind(kernel, &[src, index, out, status]) {
            set_error(format!("{e}"));
            return code_for(&e);
        }
        // the output and the status word are read-modify-write: declared Write
        let access = [
            Access::read(src.id()),
            Access::read(index.id()),
            Access::write(out.id()),
            Access::write(status.id()),
        ];
        recorded(guard.ctx.dispatch(
            kernel,
            &plan.push_constants(),
            plan.dispatch_groups(),
            1,
            &access,
        ))
    };
    if rc == ACK_OK {
        unsafe { no_duration(ms_out) };
    }
    rc
}

/// Record a grouped scatter-add into one batch (index schema 3): the index
/// kernel's KEYS pass writes every element's destination slot and position
/// into the scratch, the radix module's digit passes sort them stably by
/// slot, and the index kernel's RUNS pass adds each slot's run to its
/// destination in order. KEYS puts the scratch in the output's slot and
/// reports through the status word; the radix passes bind the index buffer,
/// the scratch and the output (only the scratch is touched); RUNS puts the
/// scratch in the index's slot, read only.
fn record_grouped(
    guard: &Inner,
    [kernel, sorter]: [&Kernel; 2],
    plan: &IndexPlan,
    [src, index, out, status]: [&Buffer; 4],
    scratch: &Buffer,
) -> i32 {
    let [keys, runs] = match plan.passes() {
        [a, b] => [a, b],
        _ => {
            set_error("index: a grouped plan without its KEYS and RUNS passes");
            return ACK_ERR_SHAPE;
        }
    };
    if let Err(e) = guard.ctx.bind(kernel, &[src, index, scratch, status]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    let rc = recorded(guard.ctx.dispatch(
        kernel,
        &keys.push_constants(),
        keys.dispatch_groups(),
        1,
        &[
            Access::read(src.id()),
            Access::read(index.id()),
            Access::write(scratch.id()),
            Access::write(status.id()),
        ],
    ));
    if rc != ACK_OK {
        return rc;
    }
    if let Err(e) = guard.ctx.bind(sorter, &[index, scratch, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    for pass in plan.radix_passes() {
        let rc = recorded(guard.ctx.dispatch(
            sorter,
            &pass.push_constants(),
            pass.dispatch_groups(),
            1,
            &[
                Access::read(index.id()),
                Access::write(scratch.id()),
                Access::read(out.id()),
            ],
        ));
        if rc != ACK_OK {
            return rc;
        }
    }
    if let Err(e) = guard.ctx.bind(kernel, &[src, scratch, out, status]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    recorded(guard.ctx.dispatch(
        kernel,
        &runs.push_constants(),
        runs.dispatch_groups(),
        1,
        &[
            Access::read(src.id()),
            Access::read(scratch.id()),
            Access::write(out.id()),
            Access::read(status.id()),
        ],
    ))
}

/// Record an elected scatter's three passes into one batch (index schema 3):
/// clear the scratch, elect each destination's last writer, write the
/// winners. Each dispatch binds four buffers, as the kernel's layout has: the
/// clear and elect passes put the scratch in the output's slot (the elect
/// pass reports through the status word in slot 3), and the write pass puts
/// it in slot 3, read only. Each pass's reads are ordered after the previous
/// pass's writes by the declared accesses.
fn record_elected(
    guard: &Inner,
    kernel: &Kernel,
    plan: &IndexPlan,
    [src, index, out, status]: [&Buffer; 4],
    scratch: &Buffer,
) -> i32 {
    let [clear, elect, write] = match plan.passes() {
        [a, b, c] => [a, b, c],
        _ => {
            set_error("index: an elected plan without its three passes");
            return ACK_ERR_SHAPE;
        }
    };
    let steps: [(&index_ops::IndexPass, [&Buffer; 4], [Access; 4]); 3] = [
        (
            clear,
            [src, index, scratch, status],
            [
                Access::read(src.id()),
                Access::read(index.id()),
                Access::write(scratch.id()),
                Access::read(status.id()),
            ],
        ),
        (
            elect,
            [src, index, scratch, status],
            [
                Access::read(src.id()),
                Access::read(index.id()),
                Access::write(scratch.id()),
                Access::write(status.id()),
            ],
        ),
        (
            write,
            [src, index, out, scratch],
            [
                Access::read(src.id()),
                Access::read(index.id()),
                Access::write(out.id()),
                Access::read(scratch.id()),
            ],
        ),
    ];
    for (pass, buffers, access) in steps {
        // each recorded dispatch takes its own descriptor set from the
        // kernel's arena, so rebinding between passes is safe (record_reduce)
        if let Err(e) = guard.ctx.bind(kernel, &buffers) {
            set_error(format!("{e}"));
            return code_for(&e);
        }
        let rc = recorded(guard.ctx.dispatch(
            kernel,
            &pass.push_constants(),
            pass.dispatch_groups(),
            1,
            &access,
        ));
        if rc != ACK_OK {
            return rc;
        }
    }
    ACK_OK
}

fn convert_plan_code(err: convert_ops::ConvertPlanError) -> i32 {
    match err {
        convert_ops::ConvertPlanError::AddressOverflow => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// One `ack_convert` call's request (convert/v1, ABI 13): what the caller asks
/// for, which the kit's own planner (`convert_ops`) turns into the push block
/// and the dispatch. Shapes, offsets and strides are in ELEMENTS of their own
/// tensor; `src_code` and `out_code` are `convert_ops::Code` storage codes (0
/// f32, 1 bf16, 2 f16, 3 i64, 4 i32, 5 i16, 6 i8, 7 u8, 8 f64, 9 bool). 128
/// bytes, every field naturally aligned, no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AckConvertRequest {
    pub src_code: u32,
    pub out_code: u32,
    pub ndim: u32,
    /// Zero; refused otherwise.
    pub flags: u32,
    pub shape: [u64; 4],
    pub in_offset: u64,
    pub in_strides: [u64; 4],
    pub out_offset: u64,
    pub out_strides: [u64; 4],
}

/// Convert a strided view of `src` into a view of `out` of another storage
/// type (convert/v1 at the C ABI, schema 1; ABI 13): `out[c] = T_out(src[c])`
/// over at most four dimensions, rounded exactly as c10 rounds (see
/// `kernels/convert.comp`'s header): float -> float through f32 to nearest
/// even, int -> float to nearest even from the exact value, int -> int by two's
/// complement truncation or extension, anything -> bool as nonzero, bool -> x
/// as 0 or 1. The kernel computes on the bits in 32-bit integer arithmetic, so
/// no device float mode or 64-bit shader feature enters the answer.
///
/// A FLOAT -> INTEGER CONVERSION IS REFUSED BY NAME
/// (`convert-float-to-integer`): c10's cast is undefined out of range, so
/// there is no reference answer to hold the kernel to. A 1- or 2-byte output
/// that is row-major dense from a 32-bit word is written whole words at a
/// time, and any other with 32-bit atomics (schema 2); every output view must
/// reach distinct addresses, and `out` may not be `src`'s buffer.
///
/// Refusals are by name before any dispatch: `ACK_ERR_NULL` for a null
/// request, `ACK_ERR_ABI` for a request of another size, `ACK_ERR_SIZE` for a
/// view past its buffer or the 2^31 word limit (`convert-address-overflow`),
/// and `ACK_ERR_SHAPE` for everything else `convert_ops::ConvertPlanError`
/// names; the storage codes, flags and conversion are decoded before any
/// device lookup. The module is built on the first call, so a failure to
/// build it is this call's error. A transposed read of two dimensions is
/// planned onto a second, tiled module (2026-10-03, `convert_ops::FLAG_TILED`;
/// the same bits, read in tiles), built on the first call that takes it. The
/// dispatch is RECORDED and runs at the next flush, so `ms_out` is NaN.
/// # Safety
/// `request` must point to `request_size` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_convert(
    dev: *const AckDevice,
    src: *const Buffer,
    out: *const Buffer,
    request: *const AckConvertRequest,
    request_size: usize,
    ms_out: *mut f64,
) -> i32 {
    if request.is_null() {
        return ACK_ERR_NULL;
    }
    if request_size != std::mem::size_of::<AckConvertRequest>() {
        set_error(format!(
            "convert: the request is {request_size} bytes, not this library's {}",
            std::mem::size_of::<AckConvertRequest>()
        ));
        return ACK_ERR_ABI;
    }
    // the caller promised `request_size` readable bytes, not alignment
    let r = unsafe { std::ptr::read_unaligned(request) };
    // the words that need no buffer are decoded before any device lookup
    let decoded = convert_ops::Code::try_from(r.src_code).and_then(|from| {
        let to = convert_ops::Code::try_from(r.out_code)?;
        if r.flags != 0 {
            return Err(convert_ops::ConvertPlanError::FlagsOutOfRange);
        }
        if !convert_ops::converts(from, to) {
            return Err(convert_ops::ConvertPlanError::FloatToInteger);
        }
        Ok(())
    });
    if let Err(e) = decoded {
        set_error(format!("convert: plan refused: {e}"));
        return ACK_ERR_SHAPE;
    }
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    // mutable only to build a module on its first use, which happens while no
    // buffer is borrowed from the guard
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // Planned against the buffers' storage FIRST (2026-10-03): which of the two
    // modules runs the dispatch is the plan's decision (convert_ops::FLAG_TILED).
    // Resolved one at a time so the code returned and the message left in
    // ack_last_error name the same handle.
    let bindings = {
        let src = match buffer_of(&device, &guard, src) {
            Ok(b) => b,
            Err(code) => return code,
        };
        let out = match buffer_of(&device, &guard, out) {
            Ok(b) => b,
            Err(code) => return code,
        };
        let storage = |b: &Buffer| convert_ops::Storage::new(b.id(), b.bytes);
        convert_ops::Bindings {
            input: storage(src),
            out: storage(out),
        }
    };
    let plan = match ConvertPlan::from_abi(
        r.src_code,
        r.out_code,
        r.ndim,
        r.flags,
        r.shape,
        convert_ops::View::new(r.in_offset, r.in_strides),
        convert_ops::View::new(r.out_offset, r.out_strides),
        bindings,
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("convert: plan refused: {e}"));
            return convert_plan_code(e);
        }
    };
    let tiled = plan.tiled();
    let missing = if tiled {
        guard.convert_tiled.is_none()
    } else {
        guard.convert.is_none()
    };
    if missing {
        let state: &mut Inner = &mut guard;
        let spirv = if tiled {
            SPV_CONVERT_TILED
        } else {
            SPV_CONVERT
        };
        match state
            .ctx
            .kernel(spirv, convert_ops::BINDINGS, convert_ops::PUSH_BYTES as u32)
        {
            Ok(k) if tiled => state.convert_tiled = Some(k),
            Ok(k) => state.convert = Some(k),
            Err(e) => {
                set_error(format!("convert pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    let src = match buffer_of(&device, &guard, src) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let kernel = if tiled {
        guard.convert_tiled.as_ref()
    } else {
        guard.convert.as_ref()
    }
    .expect("built above");
    if let Err(e) = guard.ctx.bind(kernel, &[src, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // a partly filled last word of a sub-word output is read-modify-write
    let access = [Access::read(src.id()), Access::write(out.id())];
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

fn intwise_plan_code(err: intwise_ops::IntwisePlanError) -> i32 {
    match err {
        intwise_ops::IntwisePlanError::AddressOverflow => ACK_ERR_SIZE,
        _ => ACK_ERR_SHAPE,
    }
}

/// One `ack_intwise` call's request (intwise/v1, ABI 14): what the caller asks
/// for, which the kit's own planner (`intwise_ops`) turns into the push block
/// and the dispatch. `op` is an `intwise_ops::IntOp`; the codes are
/// `convert_ops::Code` storage codes; `compute_code` is C, the common dtype
/// torch computes the operation in. With an integer C (schema 1) every code
/// is an integer one (3 i64, 4 i32, 5 i16, 6 i8, 7 u8, 9 bool). With C = f64
/// (8; schema 2, the binary64 module) x and y may be any storage code, the
/// output is f64 for an arithmetic operation and bool for a comparison, and
/// `scalar` and `alpha` carry an f64's bits. `flags` bit 0 makes y the `scalar`. Shapes,
/// offsets and strides are in ELEMENTS of their own tensor. 200 bytes, every
/// field naturally aligned, no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AckIntwiseRequest {
    pub op: u32,
    pub x_code: u32,
    pub y_code: u32,
    pub out_code: u32,
    pub compute_code: u32,
    pub ndim: u32,
    pub flags: u32,
    /// Zero; refused otherwise.
    pub reserved: u32,
    pub shape: [u64; 4],
    pub x_offset: u64,
    pub x_strides: [u64; 4],
    pub y_offset: u64,
    pub y_strides: [u64; 4],
    pub out_offset: u64,
    pub out_strides: [u64; 4],
    /// y when flags bit 0 is set; fill's value; arange's start. An f64's bits
    /// when C is f64.
    pub scalar: i64,
    /// add and sub's alpha; arange's step. With C = f64, the bits of +1.0 or -1.0.
    pub alpha: i64,
}

/// An integer or bool pointwise operation over strided views (intwise/v1 at
/// the C ABI, schema 1; ABI 14): `out = T_out(op_C(x, y))` over at most four
/// dimensions, with x and y wrapped to C first, exactly as torch's CPU kernels
/// compute it (see `kernels/intwise.comp`'s header): add, sub and mul modulo
/// C's width, the bitwise operations, c10's shifts, the comparisons (0 or 1),
/// neg, abs, fill and arange, and c10's integer divisions (div_trunc,
/// div_floor, remainder, fmod) by a scalar that is nonzero in C and not -1 in
/// an i32 or i64 C. A broadcast is a stride of 0.
///
/// Schema 2 (2026-09-28): with C = f64 the binary64 module computes torch's
/// float64 arithmetic in the same integer words, bit for bit as torch's x86
/// CPU does (IEEE 754 binary64, round to nearest even, subnormals kept, and
/// torch's NaN rule, where both operands are NaN y's for add and sub and x's
/// for mul and div; see `kernels/intwise.comp`'s header): add and sub (alpha
/// +1.0 or -1.0), mul, div, reciprocal, neg, abs, fill and the comparisons, x
/// and y of any storage code widened as torch casts them, the output f64 or,
/// for a comparison, bool. Its module is built on the first call with C = f64.
///
/// Schema 3 (2026-09-28): operation 24, cumsum, torch's CPU cumsum along the
/// request's LAST dimension (the row): element k of each row is the sum of
/// its elements 0..=k in an integer C, wrapped to C, bit for bit. It runs one
/// workgroup a row, a 1- or 2-byte output with atomics whatever its layout, a
/// bool C refused as torch refuses it, and a row longer than
/// `intwise_ops::SCAN_LIMIT` refused (`intwise-scan-too-long`).
///
/// Schema 4 (2026-09-28): operations 25 tril and 26 triu, torch's tril and
/// triu over the request's last two dimensions: element (.., i, j) is x's
/// where j - i <= `scalar` (tril) or >= `scalar` (triu), else 0. `scalar` is
/// the diagonal as a signed 64-bit value, never wrapped to C, and y is not
/// read. Any integer C, bool included; a caller sends another dtype as the
/// integer code of its width (the triangles move bits and compute none).
///
/// Schema 5 (2026-09-28): operations 27 rank_asc and 28 rank_desc, each
/// element's position in torch's STABLE sort of its row (the request's last
/// dimension), wrapped to an integer C. x may be any storage code, compared as
/// torch compares (floats: -0.0 equal to +0.0, every NaN equal and largest).
/// The output may not share x's buffer at all, and the row length and the
/// total work are bounded (`intwise_ops::RANK_ROW_LIMIT`, `RANK_WORK_LIMIT`;
/// `intwise-rank-too-long`).
///
/// All three handles must be live buffers; an operand the operation does not
/// read (y of a unary or scalar operation, x and y of fill and arange) is
/// bound but never read or checked, so bind the output there. The output may
/// share an input's buffer only through that input's own view and code (in
/// place). A 1- or 2-byte output that is row-major dense from a 32-bit word is
/// written whole words at a time, and any other with 32-bit atomics.
///
/// Refusals are by name before any dispatch: `ACK_ERR_NULL` for a null
/// request, `ACK_ERR_ABI` for a request of another size, `ACK_ERR_SIZE` for a
/// view past its buffer or the 2^31 word limit (`intwise-address-overflow`),
/// and `ACK_ERR_SHAPE` for everything else `intwise_ops::IntwisePlanError`
/// names; the operation, codes (with C = f64, the operation and the output),
/// flags, a bool operation torch refuses, a division's divisor and the
/// binary64 module's alpha are decoded before any device lookup. Each module
/// is built on the first call that needs it, so a failure to build it is this
/// call's error. The dispatch is RECORDED and
/// runs at the next flush, so `ms_out` is NaN.
/// # Safety
/// `request` must point to `request_size` readable bytes; `ms_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn ack_intwise(
    dev: *const AckDevice,
    x: *const Buffer,
    y: *const Buffer,
    out: *const Buffer,
    request: *const AckIntwiseRequest,
    request_size: usize,
    ms_out: *mut f64,
) -> i32 {
    if request.is_null() {
        return ACK_ERR_NULL;
    }
    if request_size != std::mem::size_of::<AckIntwiseRequest>() {
        set_error(format!(
            "intwise: the request is {request_size} bytes, not this library's {}",
            std::mem::size_of::<AckIntwiseRequest>()
        ));
        return ACK_ERR_ABI;
    }
    // the caller promised `request_size` readable bytes, not alignment
    let r = unsafe { std::ptr::read_unaligned(request) };
    // the words that need no buffer are decoded before any device lookup
    let decoded = intwise_ops::IntOp::try_from(r.op).and_then(|op| {
        let code = |word: u32| {
            convert_ops::Code::try_from(word)
                .map_err(|_| intwise_ops::IntwisePlanError::CodeOutOfRange)
        };
        let [x, y, out, c] = [
            code(r.x_code)?,
            code(r.y_code)?,
            code(r.out_code)?,
            code(r.compute_code)?,
        ];
        intwise_ops::check_codes(op, x, y, out, c)?;
        if r.flags & !intwise_ops::REQUEST_Y_SCALAR != 0 || r.reserved != 0 {
            return Err(intwise_ops::IntwisePlanError::FlagsOutOfRange);
        }
        if c == convert_ops::Code::Bool && op.refused_on_bool() {
            return Err(intwise_ops::IntwisePlanError::BoolOperation);
        }
        intwise_ops::check_divisor(
            op,
            c,
            r.flags & intwise_ops::REQUEST_Y_SCALAR != 0,
            r.scalar,
        )?;
        intwise_ops::check_alpha(op, c, r.alpha)?;
        Ok(c)
    });
    let binary64 = match decoded {
        Ok(c) => c == convert_ops::Code::F64,
        Err(e) => {
            set_error(format!("intwise: plan refused: {e}"));
            return ACK_ERR_SHAPE;
        }
    };
    let device = match device(dev) {
        Ok(d) => d,
        Err(code) => return code,
    };
    // mutable only to build the module on first use, which happens before any
    // buffer is borrowed from the guard
    let mut guard = match inner(&device) {
        Ok(g) => g,
        Err(code) => return code,
    };
    // Schema 6: a long row's rank may plan the radix module. Its module is
    // built here, before any buffer is borrowed from the guard, whenever the
    // request's words alone say the plan may take it.
    if guard.radix.is_none()
        && intwise_ops::may_plan_radix(r.op, r.ndim, r.shape, r.out_code, r.compute_code)
    {
        let state: &mut Inner = &mut guard;
        match state
            .ctx
            .kernel(SPV_RADIX, radix_ops::BINDINGS, radix_ops::PUSH_BYTES as u32)
        {
            Ok(k) => state.radix = Some(k),
            Err(e) => {
                set_error(format!("radix pipeline: {e}"));
                return code_for(&e);
            }
        }
    }
    let missing = if binary64 {
        guard.intwise_f64.is_none()
    } else {
        guard.intwise.is_none()
    };
    if missing {
        let state: &mut Inner = &mut guard;
        let spv = if binary64 {
            SPV_INTWISE_F64
        } else {
            SPV_INTWISE
        };
        match state
            .ctx
            .kernel(spv, intwise_ops::BINDINGS, intwise_ops::PUSH_BYTES as u32)
        {
            Ok(k) if binary64 => state.intwise_f64 = Some(k),
            Ok(k) => state.intwise = Some(k),
            Err(e) => {
                set_error(format!("intwise pipeline: {e}"));
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
    let out = match buffer_of(&device, &guard, out) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let storage = |b: &Buffer| convert_ops::Storage::new(b.id(), b.bytes);
    let bindings = intwise_ops::Bindings {
        x: storage(x),
        y: storage(y),
        out: storage(out),
    };
    let plan = match intwise_ops::IntwisePlan::from_abi(
        r.op,
        [r.x_code, r.y_code, r.out_code, r.compute_code],
        r.ndim,
        r.flags,
        r.shape,
        [
            convert_ops::View::new(r.x_offset, r.x_strides),
            convert_ops::View::new(r.y_offset, r.y_strides),
            convert_ops::View::new(r.out_offset, r.out_strides),
        ],
        r.scalar,
        r.alpha,
        bindings,
    ) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("intwise: plan refused: {e}"));
            return intwise_plan_code(e);
        }
    };
    if let Some(radix) = plan.radix() {
        let Some(kernel) = guard.radix.as_ref() else {
            // may_plan_radix and the planner disagree: a defect, named
            set_error("intwise: a radix plan without its module (may_plan_radix)");
            return ACK_ERR_UNSUPPORTED_OP;
        };
        // the scratch is the kit's own for this call, retired into the batch
        // at once (the reduce family's partials)
        let scratch = match guard.ctx.buffer(radix.scratch_words() * 4) {
            Ok(b) => b,
            Err(e) => {
                set_error(format!(
                    "intwise: a radix rank's scratch of {} words: {e}",
                    radix.scratch_words()
                ));
                return code_for(&e);
            }
        };
        let rc = record_radix(&guard, kernel, radix, x, out, &scratch);
        guard.ctx.destroy_buffer(scratch);
        if rc == ACK_OK {
            unsafe { no_duration(ms_out) };
        }
        return rc;
    }
    let kernel = if binary64 {
        guard.intwise_f64.as_ref()
    } else {
        guard.intwise.as_ref()
    }
    .expect("built above");
    if let Err(e) = guard.ctx.bind(kernel, &[x, y, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    // a slot the operation does not read is declared Read, which is
    // conservative; a partly filled last word of a sub-word output is
    // read-modify-write
    let access = [
        Access::read(x.id()),
        Access::read(y.id()),
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

/// Record a radix rank's passes into one batch (intwise schema 6): KEYS, a
/// histogram, scan and scatter for each digit pass, RANKS. Every dispatch
/// binds x, the scratch and the output; only RANKS writes the output, and
/// each pass's reads of the scratch are ordered after the previous pass's
/// writes by the declared accesses.
fn record_radix(
    guard: &Inner,
    kernel: &Kernel,
    plan: &radix_ops::RadixPlan,
    x: &Buffer,
    out: &Buffer,
    scratch: &Buffer,
) -> i32 {
    if let Err(e) = guard.ctx.bind(kernel, &[x, scratch, out]) {
        set_error(format!("{e}"));
        return code_for(&e);
    }
    for pass in plan.passes() {
        // the output is written by RANKS alone; the other passes declare it
        // Read, which is conservative
        let ranks = pass.mode() == radix_ops::RadixMode::Ranks;
        let access = [
            Access::read(x.id()),
            if ranks {
                Access::read(scratch.id())
            } else {
                Access::write(scratch.id())
            },
            if ranks {
                Access::write(out.id())
            } else {
                Access::read(out.id())
            },
        ];
        let rc = recorded(guard.ctx.dispatch(
            kernel,
            &pass.push_constants(),
            pass.dispatch_groups(),
            1,
            &access,
        ));
        if rc != ACK_OK {
            return rc;
        }
    }
    ACK_OK
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_bf16_output_product_takes_the_aligned_module_exactly_on_the_tile() {
        // every extent on the tile: the aligned module (2026-09-28)
        assert!(super::bf16_out_aligned(1024, 32000, 512));
        assert!(super::bf16_out_aligned(4096, 512, 512));
        assert!(super::bf16_out_aligned(128, 128, 32));
        // any extent off it: the edge module, whose gather takes any shape
        assert!(!super::bf16_out_aligned(4096, 1628, 512));
        assert!(!super::bf16_out_aligned(4096, 512, 1628));
        assert!(!super::bf16_out_aligned(3256, 512, 4096));
        assert!(!super::bf16_out_aligned(4096, 512, 48));
        assert!(!super::bf16_out_aligned(64, 128, 32));
    }

    #[test]
    fn a_product_takes_the_grouped_order_only_aligned_wide_and_where_built() {
        // the WORKGROUP ORDER rule (2026-10-01): aligned, at least 64 tile
        // columns, on a set holding its four RAST modules at RAST_AT
        let full = super::RAST_AT + 4;
        let at = |a_t, b_t| super::rast_index(true, 64 * 128, full, a_t, b_t);
        assert_eq!(at(0, 0), Some(8));
        assert_eq!(at(0, 1), Some(9));
        assert_eq!(at(1, 0), Some(10));
        assert_eq!(at(1, 1), Some(11));
        assert_eq!(super::rast_index(true, 32768, full, 0, 1), Some(9));
        // a grid of 63 tile columns (or a partial 64th) keeps the row order
        assert_eq!(super::rast_index(true, 63 * 128, full, 0, 0), None);
        assert_eq!(super::rast_index(true, 63 * 128 + 1, full, 0, 0), Some(8));
        assert_eq!(super::rast_index(true, 4096, full, 0, 0), None);
        // an edge product, or a set without RAST modules (32 lanes, or
        // ACK_MATMUL_ORDER=rows), never takes one
        assert_eq!(super::rast_index(false, 32768, full, 0, 0), None);
        assert_eq!(super::rast_index(true, 32768, super::RAST_AT, 0, 0), None);
    }

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
                | ACK_OP_MATMUL_F16
                | ACK_OP_INDEX
                | ACK_OP_CONVERT
                | ACK_OP_INTWISE
        );
        assert_eq!(info.op_schema_matmul_f16, ACK_OP_SCHEMA_MATMUL_F16);
        assert_eq!(info.op_schema_index, ACK_OP_SCHEMA_INDEX);
        assert_eq!((ACK_OP_INDEX, ACK_OP_SCHEMA_INDEX), (512, 3));
        assert_eq!(info.op_schema_convert, ACK_OP_SCHEMA_CONVERT);
        assert_eq!(info.op_schema_intwise, ACK_OP_SCHEMA_INTWISE);
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
        assert_eq!(std::mem::size_of::<AckAbiInfo>(), 96);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_matmul_f16), 80);
        // ABI 12 set ABI 11's reserved word: same offset, same size
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_index), 84);
        // ABI 13 appended the conversion family's schema and a zero word
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_convert), 88);
        assert_eq!(std::mem::offset_of!(AckAbiInfo, op_schema_intwise), 92);
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
    fn the_runtime_stats_layout_is_pinned() {
        // In no pointer table, like AckDispatchStats, and the ABI version did
        // not move for it: its size, its version and these offsets are the
        // whole handshake a ctypes mirror written for version 1 relies on.
        assert_eq!(std::mem::size_of::<AckRuntimeStats>(), 288);
        assert_eq!(std::mem::align_of::<AckRuntimeStats>(), 8);
        for (name, offset, expected) in [
            ("size", std::mem::offset_of!(AckRuntimeStats, size), 0),
            ("version", std::mem::offset_of!(AckRuntimeStats, version), 4),
            (
                "flush_reason_count",
                std::mem::offset_of!(AckRuntimeStats, flush_reason_count),
                8,
            ),
            (
                "batches_in_flight",
                std::mem::offset_of!(AckRuntimeStats, batches_in_flight),
                12,
            ),
            (
                "flushes_by_reason",
                std::mem::offset_of!(AckRuntimeStats, flushes_by_reason),
                16,
            ),
            (
                "pool_hits",
                std::mem::offset_of!(AckRuntimeStats, pool_hits),
                144,
            ),
            (
                "pool_misses",
                std::mem::offset_of!(AckRuntimeStats, pool_misses),
                152,
            ),
            (
                "pool_returns",
                std::mem::offset_of!(AckRuntimeStats, pool_returns),
                160,
            ),
            (
                "pool_declined",
                std::mem::offset_of!(AckRuntimeStats, pool_declined),
                168,
            ),
            (
                "pool_drained",
                std::mem::offset_of!(AckRuntimeStats, pool_drained),
                176,
            ),
            (
                "pool_reclaims",
                std::mem::offset_of!(AckRuntimeStats, pool_reclaims),
                184,
            ),
            (
                "pool_held_objects",
                std::mem::offset_of!(AckRuntimeStats, pool_held_objects),
                192,
            ),
            (
                "pool_held_bytes",
                std::mem::offset_of!(AckRuntimeStats, pool_held_bytes),
                200,
            ),
            (
                "staging_hits",
                std::mem::offset_of!(AckRuntimeStats, staging_hits),
                208,
            ),
            (
                "staging_misses",
                std::mem::offset_of!(AckRuntimeStats, staging_misses),
                216,
            ),
            (
                "staging_returns",
                std::mem::offset_of!(AckRuntimeStats, staging_returns),
                224,
            ),
            (
                "staging_declined",
                std::mem::offset_of!(AckRuntimeStats, staging_declined),
                232,
            ),
            (
                "staging_evicted",
                std::mem::offset_of!(AckRuntimeStats, staging_evicted),
                240,
            ),
            (
                "staging_held_pairs",
                std::mem::offset_of!(AckRuntimeStats, staging_held_pairs),
                248,
            ),
            (
                "staging_held_bytes",
                std::mem::offset_of!(AckRuntimeStats, staging_held_bytes),
                256,
            ),
            (
                "max_recorded_items",
                std::mem::offset_of!(AckRuntimeStats, max_recorded_items),
                264,
            ),
            (
                "recorded_items",
                std::mem::offset_of!(AckRuntimeStats, recorded_items),
                268,
            ),
            (
                "max_retired_bytes",
                std::mem::offset_of!(AckRuntimeStats, max_retired_bytes),
                272,
            ),
            (
                "async_flush",
                std::mem::offset_of!(AckRuntimeStats, async_flush),
                280,
            ),
            (
                "reserved",
                std::mem::offset_of!(AckRuntimeStats, reserved),
                284,
            ),
        ] {
            assert_eq!(offset, expected, "{name}");
        }
    }

    #[test]
    fn the_runtime_stats_entry_refuses_a_null_out_and_a_short_struct() {
        let mut stats: AckRuntimeStats = unsafe { std::mem::zeroed() };
        let full = std::mem::size_of::<AckRuntimeStats>();
        assert_eq!(
            unsafe { ack_runtime_stats(std::ptr::null(), std::ptr::null_mut(), full) },
            ACK_ERR_NULL,
            "the out pointer is checked before the device"
        );
        assert_eq!(
            unsafe { ack_runtime_stats(std::ptr::null(), &mut stats, full - 1) },
            ACK_ERR_ABI,
            "a short struct is refused before the device is touched"
        );
        assert_eq!(
            unsafe { ack_runtime_stats(std::ptr::null(), &mut stats, full) },
            ACK_ERR_NULL,
            "a null device handle is refused by `device`, after the layout check"
        );
    }

    #[test]
    fn every_flush_reason_is_named_through_the_c_abi_in_the_recorders_order() {
        for (i, reason) in crate::batch::ALL_FLUSH_REASONS.iter().enumerate() {
            let p = ack_flush_reason_name(i as u32);
            assert!(!p.is_null(), "slot {i} has no name");
            let name = unsafe { std::ffi::CStr::from_ptr(p) }.to_str().unwrap();
            assert_eq!(name, crate::batch::flush_reason_name(*reason), "slot {i}");
            assert_eq!(crate::batch::flush_reason_index(*reason), i, "slot {i}");
        }
        let count = crate::batch::FLUSH_REASON_COUNT as u32;
        assert!(
            ack_flush_reason_name(count).is_null(),
            "one past the last slot"
        );
        assert!(ack_flush_reason_name(u32::MAX).is_null());
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
        // schema 4 since exp, round, the comparisons, the logical
        // operations, where and bool storage (2026-09-22), 5 since the
        // SiLU-gated linear unit's operations 37-39 (2026-09-28), 6 since
        // tanh, log and xlogy, 40-42 (2026-09-28), 7 since the binary16
        // storage bits 10-13 (2026-09-29), 8 since the selections minimum,
        // maximum, relu and threshold_backward, 43-46 (2026-10-16); the pin is
        // the plan module's own, so the two cannot drift apart
        assert_eq!((ACK_OP_POINTWISE, ACK_OP_SCHEMA_POINTWISE), (8, 8));
        assert_eq!(ACK_OP_SCHEMA_POINTWISE, pointwise_ops::SCHEMA);
        assert_eq!(pointwise_ops::LAST_OP, 46);
        assert_eq!(
            ACK_OP_POINTWISE & (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32),
            0,
            "a fresh operator bit"
        );
        assert_eq!(pointwise_ops::PUSH_BYTES, 120);
        assert_eq!(ACK_ABI_VERSION, 14);
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
        // still means f32 on both sides; schema 3 since 2026-09-27: operation 3,
        // AbsMax, in the same block; schema 4 since 2026-09-29: storage bit 2,
        // the float64 sum
        assert_eq!((ACK_OP_REDUCE, ACK_OP_SCHEMA_REDUCE), (16, 4));
        assert_eq!(
            ACK_OP_REDUCE
                & (ACK_OP_MATMUL_BF16 | ACK_OP_CAST | ACK_OP_MATMUL_F32 | ACK_OP_POINTWISE),
            0,
            "a fresh operator bit"
        );
        assert_eq!(reduce_ops::PUSH_BYTES, 80);
        assert_eq!(reduce_ops::REQUEST_BYTES, 84);
        assert_eq!(reduce_ops::STORAGE_FLAGS, 0b111);
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
        flagged[80] = 0b1000;
        assert_eq!(call(flagged.as_ptr(), flagged.len()), ACK_ERR_SHAPE);
        let mut msg = [0 as c_char; 128];
        assert_eq!(unsafe { ack_last_error(msg.as_mut_ptr(), 128) }, ACK_OK);
        let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) }.to_string_lossy();
        assert!(text.contains("reduce-storage-flags-out-of-range"), "{text}");
        // and each of the five ADMITTED words reaches the null device handle
        // instead, which is what makes the refusal above about the flags word
        // rather than about the longer block
        for word in [0u32, 0b01, 0b10, 0b11, 0b100] {
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
            (&[(0usize, 4u32)][..], "reduce-operation-out-of-range"), // 3 is AbsMax since 2026-09-27
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
    fn the_f16_operator_has_its_bit_and_schema_and_refuses_as_the_bf16_one_does() {
        assert_eq!((ACK_OP_MATMUL_F16, ACK_OP_SCHEMA_MATMUL_F16), (256, 1));
        assert_eq!(
            ACK_OP_MATMUL_F16
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE
                    | ACK_OP_LOSS
                    | ACK_OP_EMBEDDING),
            0,
            "a fresh operator bit"
        );
        assert_eq!(
            (
                ACK_FEATURE_SHADER_FLOAT16,
                ACK_FEATURE_COOPMAT_F16_16X16X16_SUBGROUP
            ),
            (128, 256)
        );
        // the same refusals, in the same order, before any device exists: a
        // zero dimension by shape, then a device handle that names nothing
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let mut ms = 0.0f64;
        for (m, n, k) in [(0, 128, 128), (128, 0, 128), (128, 128, 0), (128, 128, 128)] {
            let bf16 = unsafe { ack_matmul_bf16(dev, none, none, none, m, n, k, 0, 0, &mut ms) };
            let f16 = unsafe { ack_matmul_f16(dev, none, none, none, m, n, k, 0, 0, &mut ms) };
            assert_eq!(bf16, f16, "{m}x{n}x{k}");
            let bf16_z = unsafe {
                ack_matmul_bf16_z(
                    dev, none, none, none, m, n, k, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &mut ms,
                )
            };
            let f16_z = unsafe {
                ack_matmul_f16_z(
                    dev, none, none, none, m, n, k, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &mut ms,
                )
            };
            assert_eq!(bf16_z, f16_z, "z {m}x{n}x{k}");
            if m == 0 || n == 0 || k == 0 {
                assert_eq!(f16, ACK_ERR_SHAPE);
            } else {
                assert_ne!(f16, ACK_OK, "a null device is refused");
            }
        }
    }

    #[test]
    fn the_index_operator_has_its_bit_and_schema_and_refuses_a_block_before_a_device() {
        assert_eq!((ACK_OP_INDEX, ACK_OP_SCHEMA_INDEX), (512, 3));
        assert_eq!(ACK_OP_SCHEMA_INDEX, index_ops::SCHEMA);
        assert_eq!(
            ACK_OP_INDEX
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE
                    | ACK_OP_LOSS
                    | ACK_OP_EMBEDDING
                    | ACK_OP_MATMUL_F16),
            0,
            "a fresh operator bit"
        );
        assert_eq!(index_ops::PUSH_BYTES, 112);
        assert_eq!(
            index_ops::BINDINGS,
            4,
            "the Vulkan baseline's storage buffers per stage"
        );
        // the request's layout, which ack.py and the torch adapter mirror
        assert_eq!(std::mem::size_of::<AckIndexRequest>(), 200);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, shape), 24);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, index_offset), 56);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, src_offset), 96);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, out_offset), 136);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, bound), 176);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, value), 184);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, alpha_bits), 192);
        assert_eq!(std::mem::offset_of!(AckIndexRequest, reserved), 196);
        // a null request, a request of another size, a nonzero reserved word
        // and an unknown operation or storage refuse before the device lookup
        // (no device is open here)
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let mut ms = 0.0f64;
        let size = std::mem::size_of::<AckIndexRequest>();
        let rc = unsafe { ack_index(dev, none, none, none, none, std::ptr::null(), size, &mut ms) };
        assert_eq!(rc, ACK_ERR_NULL);
        let good: AckIndexRequest = unsafe { std::mem::zeroed() };
        let rc = unsafe { ack_index(dev, none, none, none, none, &good, size - 8, &mut ms) };
        assert_eq!(rc, ACK_ERR_ABI);
        for (op, dtype, reserved) in [(3, 0, 0), (0, 9, 0), (0, 0, 1)] {
            let r = AckIndexRequest {
                op,
                dtype,
                reserved,
                ..good
            };
            let rc = unsafe { ack_index(dev, none, none, none, none, &r, size, &mut ms) };
            assert_eq!(
                rc, ACK_ERR_SHAPE,
                "op {op} dtype {dtype} reserved {reserved}"
            );
        }
        // schema 2's atomic output is the planner's to choose: a request
        // carrying the flag is refused with the unknown ones, also before the
        // device lookup (schema 1 decoded flags only after it)
        for flags in [index_ops::FLAG_ATOMIC_OUT, 32] {
            let r = AckIndexRequest { flags, ..good };
            let rc = unsafe { ack_index(dev, none, none, none, none, &r, size, &mut ms) };
            assert_eq!(rc, ACK_ERR_SHAPE, "flags {flags}");
        }
        // well-formed words reach the device lookup, which a null handle fails
        let rc = unsafe { ack_index(dev, none, none, none, none, &good, size, &mut ms) };
        assert_ne!(rc, ACK_OK);
        assert_ne!(rc, ACK_ERR_SHAPE);
    }

    #[test]
    fn the_convert_operator_has_its_bit_and_schema_and_refuses_a_block_before_a_device() {
        assert_eq!((ACK_OP_CONVERT, ACK_OP_SCHEMA_CONVERT), (1024, 2));
        assert_eq!(ACK_OP_SCHEMA_CONVERT, convert_ops::SCHEMA);
        assert_eq!(
            ACK_OP_CONVERT
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE
                    | ACK_OP_LOSS
                    | ACK_OP_EMBEDDING
                    | ACK_OP_MATMUL_F16
                    | ACK_OP_INDEX),
            0,
            "a fresh operator bit"
        );
        assert_eq!(convert_ops::PUSH_BYTES, 72);
        assert_eq!(convert_ops::BINDINGS, 2);
        // the request's layout, which ack.py and the torch adapter mirror
        assert_eq!(std::mem::size_of::<AckConvertRequest>(), 128);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, flags), 12);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, shape), 16);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, in_offset), 48);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, in_strides), 56);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, out_offset), 88);
        assert_eq!(std::mem::offset_of!(AckConvertRequest, out_strides), 96);
        // a null request, a request of another size, an unknown storage code,
        // a nonzero flags word and a float -> integer conversion refuse before
        // the device lookup (no device is open here)
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let mut ms = 0.0f64;
        let size = std::mem::size_of::<AckConvertRequest>();
        let rc = unsafe { ack_convert(dev, none, none, std::ptr::null(), size, &mut ms) };
        assert_eq!(rc, ACK_ERR_NULL);
        let good: AckConvertRequest = unsafe { std::mem::zeroed() };
        let rc = unsafe { ack_convert(dev, none, none, &good, size - 8, &mut ms) };
        assert_eq!(rc, ACK_ERR_ABI);
        for (from, to, flags) in [
            (10, 0, 0),
            (0, 10, 0),
            (0, 1, 1),
            (0, 3, 0),
            (8, 7, 0),
            (2, 5, 0),
        ] {
            let r = AckConvertRequest {
                src_code: from,
                out_code: to,
                flags,
                ..good
            };
            let rc = unsafe { ack_convert(dev, none, none, &r, size, &mut ms) };
            assert_eq!(rc, ACK_ERR_SHAPE, "from {from} to {to} flags {flags}");
        }
        // well-formed words reach the device lookup, which a null handle fails
        for (from, to) in [(0, 0), (3, 0), (2, 9), (9, 8), (7, 6)] {
            let r = AckConvertRequest {
                src_code: from,
                out_code: to,
                ..good
            };
            let rc = unsafe { ack_convert(dev, none, none, &r, size, &mut ms) };
            assert_ne!(rc, ACK_OK);
            assert_ne!(rc, ACK_ERR_SHAPE, "from {from} to {to} decodes");
        }
    }

    #[test]
    fn the_intwise_operator_has_its_bit_and_schema_and_refuses_a_request_before_a_device() {
        assert_eq!((ACK_OP_INTWISE, ACK_OP_SCHEMA_INTWISE), (2048, 6));
        assert_eq!(ACK_OP_SCHEMA_INTWISE, intwise_ops::SCHEMA);
        assert_eq!(
            ACK_OP_INTWISE
                & (ACK_OP_MATMUL_BF16
                    | ACK_OP_CAST
                    | ACK_OP_MATMUL_F32
                    | ACK_OP_POINTWISE
                    | ACK_OP_REDUCE
                    | ACK_OP_ROWWISE
                    | ACK_OP_LOSS
                    | ACK_OP_EMBEDDING
                    | ACK_OP_MATMUL_F16
                    | ACK_OP_INDEX
                    | ACK_OP_CONVERT),
            0,
            "a fresh operator bit"
        );
        assert_eq!(intwise_ops::PUSH_BYTES, 108);
        assert_eq!(intwise_ops::BINDINGS, 3);
        // the request's layout, which ack.py and the torch adapter mirror
        assert_eq!(std::mem::size_of::<AckIntwiseRequest>(), 200);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, compute_code), 16);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, reserved), 28);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, shape), 32);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, x_offset), 64);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, y_offset), 104);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, out_offset), 144);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, out_strides), 152);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, scalar), 184);
        assert_eq!(std::mem::offset_of!(AckIntwiseRequest, alpha), 192);
        // a null request, a request of another size, an unknown operation or
        // code, a float code where C is an integer, an operation C's module
        // does not have, an f64 C's wrong output or alpha, a nonzero reserved
        // or unknown flag, a bool operation torch refuses and a division's
        // divisor all refuse before the device lookup (no device is open here)
        let dev: *const AckDevice = std::ptr::null();
        let none: *const Buffer = std::ptr::null();
        let mut ms = 0.0f64;
        let size = std::mem::size_of::<AckIntwiseRequest>();
        let rc = unsafe { ack_intwise(dev, none, none, none, std::ptr::null(), size, &mut ms) };
        assert_eq!(rc, ACK_ERR_NULL);
        let good = AckIntwiseRequest {
            op: 0,
            x_code: 3,
            y_code: 3,
            out_code: 3,
            compute_code: 3,
            ndim: 1,
            flags: 0,
            reserved: 0,
            shape: [4, 0, 0, 0],
            x_offset: 0,
            x_strides: [1, 0, 0, 0],
            y_offset: 0,
            y_strides: [1, 0, 0, 0],
            out_offset: 0,
            out_strides: [1, 0, 0, 0],
            scalar: 0,
            alpha: 1,
        };
        let rc = unsafe { ack_intwise(dev, none, none, none, &good, size - 8, &mut ms) };
        assert_eq!(rc, ACK_ERR_ABI);
        for bad in [
            AckIntwiseRequest { op: 22, ..good },
            // a division by a tensor, by 0, and by -1 in an i64 C
            AckIntwiseRequest { op: 18, ..good },
            AckIntwiseRequest {
                op: 19,
                flags: 1,
                scalar: 0,
                ..good
            },
            AckIntwiseRequest {
                op: 20,
                flags: 1,
                scalar: -1,
                ..good
            },
            AckIntwiseRequest { x_code: 10, ..good },
            AckIntwiseRequest { y_code: 0, ..good },
            AckIntwiseRequest {
                compute_code: 8,
                ..good
            },
            AckIntwiseRequest { flags: 2, ..good },
            AckIntwiseRequest {
                reserved: 1,
                ..good
            },
            AckIntwiseRequest {
                op: 1,
                compute_code: 9,
                ..good
            },
            // C = f64: alpha the integer 1 rather than 1.0's bits; an integer
            // operation; a comparison written to f64
            AckIntwiseRequest {
                compute_code: 8,
                out_code: 8,
                ..good
            },
            AckIntwiseRequest {
                op: 3,
                compute_code: 8,
                out_code: 8,
                alpha: intwise_ops::F64_PLUS_ONE,
                ..good
            },
            AckIntwiseRequest {
                op: 10,
                compute_code: 8,
                out_code: 8,
                ..good
            },
            // the scan (schema 3) on a bool C, which torch refuses, and on an
            // f64 C, which is not the binary64 module's; and an operation past it
            AckIntwiseRequest {
                op: 24,
                x_code: 9,
                y_code: 9,
                out_code: 9,
                compute_code: 9,
                ..good
            },
            AckIntwiseRequest {
                op: 24,
                compute_code: 8,
                out_code: 8,
                ..good
            },
            AckIntwiseRequest { op: 29, ..good },
            // a rank (schema 5) into an f32 output, which must be an integer code
            AckIntwiseRequest {
                op: 27,
                x_code: 0,
                out_code: 0,
                ..good
            },
            // the triangles (schema 4) with C = f64: not the binary64 module's
            AckIntwiseRequest {
                op: 25,
                compute_code: 8,
                out_code: 8,
                ..good
            },
        ] {
            let rc = unsafe { ack_intwise(dev, none, none, none, &bad, size, &mut ms) };
            assert_eq!(rc, ACK_ERR_SHAPE, "{bad:?}");
        }
        // well-formed words reach the device lookup, which a null handle fails
        for ok in [
            good,
            AckIntwiseRequest { op: 17, ..good },
            AckIntwiseRequest {
                op: 21,
                flags: 1,
                scalar: -3,
                ..good
            },
            AckIntwiseRequest {
                op: 8,
                out_code: 9,
                ..good
            },
            AckIntwiseRequest {
                op: 2,
                compute_code: 9,
                flags: 1,
                ..good
            },
            // C = f64 (schema 2): an f32 x plus an i64 y with alpha -1.0, a
            // division (alpha unread), a comparison to bool, and fill
            AckIntwiseRequest {
                x_code: 0,
                compute_code: 8,
                out_code: 8,
                alpha: intwise_ops::F64_MINUS_ONE,
                ..good
            },
            AckIntwiseRequest {
                op: 22,
                compute_code: 8,
                out_code: 8,
                ..good
            },
            AckIntwiseRequest {
                op: 10,
                x_code: 2,
                y_code: 9,
                compute_code: 8,
                out_code: 9,
                ..good
            },
            AckIntwiseRequest {
                op: 16,
                compute_code: 8,
                out_code: 8,
                scalar: -1,
                ..good
            },
            // the scan (schema 3): a bool x summed in an i64 C
            AckIntwiseRequest {
                op: 24,
                x_code: 9,
                ..good
            },
            // the triangles (schema 4): a bool tril, and a triu whose diagonal
            // is far outside every C but i64 (it is not wrapped)
            AckIntwiseRequest {
                op: 25,
                x_code: 9,
                y_code: 9,
                out_code: 9,
                compute_code: 9,
                scalar: -3,
                ..good
            },
            AckIntwiseRequest {
                op: 26,
                x_code: 5,
                y_code: 5,
                out_code: 5,
                compute_code: 5,
                scalar: 1 << 40,
                ..good
            },
            // the ranks (schema 5): an f32 and an f64 x ranked into i64
            AckIntwiseRequest {
                op: 27,
                x_code: 0,
                ..good
            },
            AckIntwiseRequest {
                op: 28,
                x_code: 8,
                ..good
            },
        ] {
            let rc = unsafe { ack_intwise(dev, none, none, none, &ok, size, &mut ms) };
            assert_ne!(rc, ACK_OK);
            assert_ne!(rc, ACK_ERR_SHAPE, "{ok:?} decodes");
        }
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
        // The embedding family joined ABI 9 without bumping it. The CURRENT
        // version is pinned once, in the ABI test; a second pin here said 11
        // through the index family's bump to 12 and failed only under the
        // verify's fault-injection build (2026-09-27).
        const _: () = assert!(
            ACK_ABI_VERSION >= 9,
            "the embedding family is part of every ABI from 9"
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
