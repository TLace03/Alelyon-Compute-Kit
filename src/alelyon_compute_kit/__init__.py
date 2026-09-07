"""Alelyon Compute Kit's side-effect-free capability and registration API.

Importing this package does not load drivers, discover plugins, allocate device
memory, or establish that a training workload is supported. Capability matches
describe supplied declarations; execution and validation are separate layers.
"""

from .backends import (
    BACKEND_API_VERSION,
    BackendDescriptor,
    BackendRegistry,
    DeviceRejection,
    RegistryError,
    SelectionResult,
)
from .capabilities import (
    CapabilityError,
    CapabilityMatch,
    DeviceCapabilities,
    DeviceIdentity,
    DType,
    EvidenceStatus,
    NamedLimit,
    OperatorCapability,
    OperatorRequirement,
    Precision,
    ValidationEvidence,
    match_capabilities,
)

__version__ = "0.1.0a1"

__all__ = [
    "BACKEND_API_VERSION",
    "BackendDescriptor",
    "BackendRegistry",
    "CapabilityError",
    "CapabilityMatch",
    "DeviceCapabilities",
    "DeviceIdentity",
    "DeviceRejection",
    "DType",
    "EvidenceStatus",
    "NamedLimit",
    "OperatorCapability",
    "OperatorRequirement",
    "Precision",
    "RegistryError",
    "SelectionResult",
    "ValidationEvidence",
    "match_capabilities",
    "__version__",
]
