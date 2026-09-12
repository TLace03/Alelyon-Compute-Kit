# Alelyon Compute Kit

Alelyon Compute Kit is an experimental vendor-neutral compute SDK for training
and general computation. It combines explicit device/operator contracts with a
Vulkan implementation. Support depends on device features and workload tests.

The PyPI distribution is **`alelyon-ai`**; the Python namespace is
**`alelyon_compute_kit`**. This repository uses the Apache License 2.0.
Previously published `0.1.0a0` and `0.1.0a1` artifacts remain MIT-licensed.

## Current package

Version `0.1.0a3` adds a Windows AMD64 native library, reviewed Rust and shader
sources, Python buffers and an experimental vector-coded expert API. The root
import retains the immutable capability records and explicit backend registry:
it does not load drivers, discover plugins, compile code or allocate a device.
No backend is automatically registered.

The runtime includes matrix, pointwise, reduction, normalization, loss and
embedding operations. Its optional `vq4/v1` extension provides encoding,
decoding, direct packed matrix products and AdamW updates from compressed
weights, gradients and optimizer moments. It uses a four-bit index for a
vector of 8, 16 or 32 values, with a 16-entry FP32 codebook per tensor.

The rate includes codebooks, padding and metadata: a 256 by 256 tensor with
group 16 occupies 3,128 serialized bytes, or 0.3818359375 bits per value.
Small tensors can exceed one bit per value. Coded logical values share a
codebook and do not represent the same independent degrees of freedom as
full-precision parameters. Arithmetic and temporary outputs remain FP32;
codebook fitting and diagnostics use CPU arrays. Re-encoding is lossy.

This is a primitive compute interface, not a complete CUDA replacement or a
ready-to-train language model. Current native device measurements are from one
AMD Radeon RX 9070 XT. Other vendors, platforms, trillion-parameter training,
a 48-hour training target and comparative performance remain unmeasured.

## Installation and an explicit device operation

Use 64-bit Windows and Python 3.10 or later, with a compatible Vulkan driver:

```console
python -m pip install "alelyon-ai[numpy]==0.1.0a3"
```

```python
import numpy as np
from alelyon_compute_kit import ack, vq

values = np.zeros((256, 256), dtype=np.float32)
with ack.Device() as device:
    backend = vq.VQDevice(device)
    image = backend.encode(values, vq.fit_codebook(values), group=16)
    restored = backend.decode(image)
    print(image.layout)
```

`Device()` explicitly opens Vulkan and refuses missing drivers or incompatible
features. `VQDevice` requires the optional schema. Convenience operations upload
inputs and download outputs; they do not retain a whole graph on the GPU.
No compiler or download runs during installation or device construction.

`VQAdamW` stores encoded weights and both moments, re-encoding each update.
`ExpertBank` persists complete expert generations through hashed shards and an
atomic pointer. `GainScheduler` prioritizes measured held-out improvement per
second while reserving exploration. Bank capacity, materialized weights and
actually updated weights are separate quantities. Declaring a large bank does
not initialize or train it.

## Run or update an expert without ROCm

The model-facing harness uses the native Vulkan path directly. It reads a
materialized expert from an `ExpertBank`, encodes input activations, runs packed
matrix products, and can commit one clipped AdamW update with weights and both
moments in the same generation. It does not import ROCm, CUDA, or a framework.

Create a three-shard expert bank once:

```python
from alelyon_compute_kit import expert_bank

bank = expert_bank.ExpertBank.create(
    "./my-bank",
    expert_bank.BankConfig(1, 1, ((128, 128),) * 3, group=16),
)
bank.initialize_expert(0, 0, seed=7, zero_optimizer_state=True)
```

Then run a forward pass or a durable training step from `.npy` arrays:

```console
python -m alelyon_compute_kit.harness --device --bank ./my-bank ^
  --input ./batch.npy --output ./prediction.npy

python -m alelyon_compute_kit.harness --device --bank ./my-bank ^
  --input ./batch.npy --target ./target.npy --steps 4 ^
  --output ./prediction.npy --force
```

`--device` is required so opening Vulkan is visible and explicit. The harness
uses CPU codebook calibration and diagnostics, while packed matmul and AdamW
run on the selected Vulkan device. It is a linear-expert adapter: model owners
still supply tokenisation, routing, graph composition and checkpoint policy.
The harness does not establish model quality or trillion-parameter capacity.

## Capability matching

This fabricated declaration demonstrates matching without hardware discovery:

```python
from alelyon_compute_kit import (
    BackendDescriptor, BackendRegistry, DeviceCapabilities, DeviceIdentity,
    DType, OperatorCapability, OperatorRequirement, Precision,
)

registry = BackendRegistry()
registry.register(BackendDescriptor(
    backend_id="example",
    name="Example declaration",
    devices=(DeviceCapabilities(
        identity=DeviceIdentity("example", "device-0", "Example device"),
        operators=(OperatorCapability("mm", 1, DType.FLOAT32, Precision.FP32),),
    ),),
))
selection = registry.select((
    OperatorRequirement("mm", 1, DType.FLOAT32, Precision.FP32),
))
assert len(selection.matches) == 1
```

A match checks supplied declarations. It does not verify hardware, numerical
correctness or training readiness. Unsupported requirements retain named reasons;
selection does not silently change precision or choose a fallback.

## Development

Native building is explicit; installing the source distribution without a
separately built matching DLL refuses instead of invoking Cargo automatically.
See [native packaging](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/docs/NATIVE_PACKAGING.md) for build commands and source
manifest rules, [architecture](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/docs/ARCHITECTURE.md) for extension boundaries,
and [release instructions](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/docs/RELEASING.md) for the manual PyPI workflow.

The excluded private framework adapter and research models are not part of this
package. The included legacy `torch_ops` module is separately imported, uses
host copies, and is not the full framework backend.
