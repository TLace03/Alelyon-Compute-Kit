# Alelyon Compute Kit

Alelyon Compute Kit is being developed as a vendor-neutral compute SDK for
training and general computation. The goal is one programming interface with
backends for different device architectures, compiler targets and driver APIs.
Support is determined by explicit capabilities and workload tests.

The PyPI distribution name is **`alelyon-ai`**. The Python namespace is
**`alelyon_compute_kit`**. This repository uses the [MIT license](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/LICENSE).

## Current state

This initial standalone package contains immutable capability records and an
explicit backend registry. It has no runtime dependencies and does not load a
driver, import plugins, probe hardware or allocate device memory on import.

The native Vulkan runtime, kernels and experimental PyTorch integration are
being prepared for a bounded SDK export from their existing development tree.
They are not included in this initial package. No device backend is registered
by default. This version cannot yet run training or replace an installed CUDA
toolchain. A published release will identify its actual backend, architecture,
operator, precision and platform coverage.

## Capability matching

The following is a fabricated declaration illustrating the API; it performs no
hardware discovery or computation:

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

A match means the supplied declaration meets the supplied requirements. It does
not verify hardware availability, numerical correctness or training readiness.
Evidence records remain unverified references and do not change matching.
Unsupported requirements return named reasons; selection neither ranks vendors
nor silently changes precision or falls back to a different device.

## Development

Use an isolated Python environment:

```console
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

The current source targets Python 3.10 and later. A pure-Python wheel's platform
tag describes these capability modules, not native accelerator support. Native
packages will require separate platform and installation checks.

[Architecture and extension boundaries](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/docs/ARCHITECTURE.md) describe the
planned runtime layers and the acceptance required before backend adoption.
The [release guide](https://github.com/TLace03/Alelyon-Compute-Kit/blob/main/docs/RELEASING.md) describes the manually dispatched
validation and PyPI publishing workflow. Publishing is an explicit option;
ordinary pushes do not release a package.
