# SDK architecture and extension boundaries

The SDK separates device declarations, execution and workload evidence. An
operator is an exact schema, dtype, arithmetic policy and layout contract;
unsupported combinations are refused before dispatch.

| Layer | Current implementation |
|---|---|
| Capability registry | Immutable supplied declarations; no discovery or device execution |
| Device runtime | Vulkan queues, buffers, ownership, transfers, synchronization and terminal failures |
| Kernels | Versioned plans and SPIR-V modules with shape, stride and byte-capacity checks |
| Vector-coded tensors | Optional VQ schema 1, Python images, packed matmul and compressed-state AdamW |
| Expert persistence and selection | Atomic local generations and held-out gain-per-second scheduling |
| Framework integration | Legacy optional host-copy adapter; complete framework backend excluded |
| Complete model training and serving | Requires model, routing, graph, checkpoint and workload integration |

The capability registry does not automatically turn a declaration into an
executable backend. Explicitly importing `ack` makes the binding available;
constructing `ack.Device` opens a device. The base package remains free of
implicit network, compiler and driver activity.

## Portability and extension

Vulkan is the implemented native path. The published native wheel targets
Windows AMD64. Other platforms require separately built and tested artifacts.
Metal, DirectX, WebGPU, CPU and CUDA/HIP compatibility are possible future
backends, not implemented universal support. A chip with no suitable driver or
compiler target requires additional implementation.

The C interface currently requires ABI 10. Optional VQ support is discovered
through `ack_vq_schema`, independently of the existing ABI layout. New schemas
must preserve closed admission and explicit unsupported states. Capability
matching must not invent combinations from separately supported variants.

## Fractional storage and bounded execution

VQ uses a nibble index into 16 FP32 vectors. Group sizes 8, 16 and 32 amortize
the index across logical tensor values. Full codebooks, padded words and wire
metadata count toward storage. Sharing vector entries restricts representational
freedom; logical capacity is not independent full-precision capacity.

The native extension executes encode, decode, packed-pair matmul and AdamW
from four encoded states into three FP32 scratch arrays. Convenience bindings
transfer operands/results per call, and CPU codebook calibration follows an
optimizer update. There is no claim of a GPU-resident graph or sub-bit arithmetic.
Each call owns its buffers, checks device status and releases resources on failure.

Active tensors are bounded to 2^28 values. Three-array AdamW scratch must fit
that bound. Matrix dimensions are at most 65,536; the scalar packed kernel
admits at most 2^28 multiply-accumulates in one dispatch. Larger products need
explicit bounded chunks. No complete mixture-of-experts token router is supplied.

## Expert generations and value measurements

`ExpertBank.create` writes configuration, not every potential expert. Initializing
an expert writes deterministic compressed random tensors without a dense full
bank allocation. Each update writes unique shards, then atomically replaces one
expert pointer. Weights, moments and optimizer-step metadata can commit together.
Readers use one committed pointer snapshot. Incomplete and old generations are
retained; automatic garbage collection is not implemented.

The bank assumes a trusted local single writer and concurrent readers. Checksums
detect changed bytes but do not authenticate a hostile same-user writer. File
fsync and pointer replacement address process interruption; Windows directory
power-loss durability is unmeasured. There is no multi-bank transaction. Run
aggregate statistics with writers quiescent; file-byte counts include historical
and orphan generations and do not measure filesystem allocated blocks.

The gain scheduler compares positive held-out loss reduction per elapsed second
and reserves exploration. Comparable tasks, sample counts and timing scopes are
the caller's responsibility. A score is a measurement, not a promise of future
expert usefulness. Potential, materialized, active and updated counts must remain
separate; cumulative update applications are not unique parameter coverage.

## Validation and package boundary

Kernel tests cover outputs, tails, transposes, ownership, nonfinite status and
refusals on their declared domains. Lossy optimizer quality and convergence
require separate experiments. Full training acceptance needs the complete
forward/backward/update graph, state and RNG restoration, routing, residency and
evaluation. Performance comparisons need matching workloads and completion
criteria, synchronized timing and separate setup/checkpoint costs.

The public export is an explicit source allowlist with hashes, shader provenance,
locked dependencies and license notices. It excludes private application Runtime,
trading, hosted identity, signing, research models, datasets and checkpoints.
Manifests are declared traceability, not authenticated provenance. Native wheel,
source archive, DLL and shaders have distinct checks. Publication is a separate,
explicit action after artifact and clean-install acceptance.
