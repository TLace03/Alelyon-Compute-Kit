# SDK architecture and extension boundaries

The SDK is organized around explicit contracts, not a list of chip vendors.
A backend must identify an actual device and the exact operations, storage
types, arithmetic policies and limits it supports. Unsupported requirements
must remain visible before allocating model state or starting a training step.

## Layers

| Layer | Responsibility | Current standalone status |
|---|---|---|
| Capability and backend registry | Immutable device/operator declarations, version checks, requirement matching, named refusals | Implemented; no driver execution |
| Device runtime | Contexts, queues, buffers, ownership, transfers, synchronization, terminal failure | Vulkan source integration pending |
| Kernel implementation | Versioned bindings, shape/stride contracts, compiler targets and numerical policies | Vulkan source integration pending |
| Framework integration | Tensor views, autograd, optimizer state, RNG, serialization and execution accounting | Experimental integration pending |
| Workload admission | Complete updates, fresh-process resume, residency and stability | Pending standalone validation |
| Backend optimization | Device-specific accelerated paths under the same declared semantics | Requires per-device measurements |

The initial registry holds declarations supplied explicitly by the caller. It
does not execute plugin callbacks or discover entry points. A future executable
backend interface must version that additional authority separately. Importing
the SDK must remain free of implicit device, network and compiler activity.

## Portability

Vulkan is the first native implementation path. Other driver/compiler targets
can be added without changing device selection into vendor-name conditionals.
Future targets may include Metal, DirectX, WebGPU, CPU implementations and
optional compatibility with CUDA or HIP libraries. These are extension targets,
not currently implemented or tested standalone backends.

Architectures with no compatible driver or compiler target need a new backend.
An API abstraction cannot supply missing device instructions or memory. Device
capabilities must include relevant compiler, driver and arithmetic behavior;
matching a brand or CPU instruction set is insufficient.

An operator capability is an exact schema/dtype/precision variant. Combining
separate records must not invent a Cartesian product of supported variants.
Limits have names and units. Future limits, layouts and execution contracts
must retain explicit versioning and refusal for unsupported requirements.

## Correctness and performance

Kernel validation checks complete outputs, tails, strides, alias/ownership
refusals, deterministic repeats and nonfinite behavior on its declared domain.
FP32 alone does not describe accumulation order, contraction, denormal treatment
or rounding. An optimized path must declare and satisfy its numerical contract.

Training admission additionally checks forward/backward, every gradient and
update, optimizer slots, tied parameters, accumulation, clipping, RNG and exact
same-backend resume in a fresh process. Transfer and per-operator fallback
accounting must be coherent and bound to a fresh execution identity. Resource
ownership failures remain terminal evidence across counter resets.

Performance comparisons require matching model, data, precision and completion
criteria, synchronized timings, repeated fresh processes, and separate setup,
training, diagnostics and checkpoint costs. Device-specific speed does not
establish a universal advantage over every available backend.

## Public package boundary

The SDK export will contain only explicitly inventoried compute implementation,
its build inputs and focused tests/examples. It will exclude private trading,
identity/signing, hosted-service, research-model, dataset and checkpoint code.
Every exported source file must have a source revision/digest and reviewed
license/dependency provenance. Build-only helpers must not import the private
application Runtime.

The package keeps source-distribution, Python wheel, native library and shader
artifacts distinct. Native wheels must name their actual platform/architecture
and include required licenses. A clean installation must refuse unavailable
drivers or incompatible ABIs without compiling, downloading or changing the
machine implicitly. Release uploads and credentials are separate from local
builds and tests.
