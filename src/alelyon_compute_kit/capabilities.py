"""Versioned declarations for capability matching, without hardware discovery.

These records describe what a caller declares. Neither names, operator entries,
nor attached evidence establish hardware availability or training readiness.
Evidence is an unverified reference; this module never opens it. Operator schemas
define tensor roles, layouts and semantics. A dtype names the principal payload
type, and each precision variant is declared separately. There is no tensor,
stream, allocation or execution interface in this discovery contract.
Scalar strings must be exact built-in strings; retaining subclasses would also
retain caller-defined equality/hash behavior inside otherwise immutable records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import islice
import re

CAPABILITY_SCHEMA_VERSION = 1
MAX_QUANTITY = (1 << 63) - 1
MAX_OPERATORS = 4096
MAX_LIMITS = 64
MAX_EVIDENCE = 128


class CapabilityError(ValueError):
    """A malformed declaration; ``code`` is a stable, named refusal."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DType(str, Enum):
    BOOL = "bool"
    INT32 = "int32"
    INT64 = "int64"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class Precision(str, Enum):
    """Declared arithmetic modes, not numerical error or determinism guarantees.

Single-type modes name operand and accumulator arithmetic. Mixed modes name
operand arithmetic followed by accumulator arithmetic. INTEGER names integer
arithmetic; the operator schema must define overflow behavior. A mode does not
promise correct rounding, denormal preservation or a validated implementation.
    """

    INTEGER = "integer"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"
    FP16_FP32 = "fp16-fp32"
    BF16_FP32 = "bf16-fp32"
    TF32_FP32 = "tf32-fp32"


class EvidenceStatus(str, Enum):
    UNMEASURED = "unmeasured"
    PASSED = "passed"
    FAILED = "failed"
    REFUSED = "refused"


def _text(value: object, label: str, *, maximum: int = 128) -> str:
    if (type(value) is not str or not value or len(value) > maximum
            or value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise CapabilityError(f"invalid-{label}")
    return value


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value):
        raise CapabilityError(f"invalid-{label}")
    return value


def _quantity(value: object, label: str, *, minimum: int = 0,
              maximum: int = MAX_QUANTITY) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CapabilityError(f"invalid-{label}")
    return value


def _version(value: object, label: str) -> int:
    return _quantity(value, label, minimum=1, maximum=(1 << 31) - 1)


def _enum(value: object, enum_type: type[Enum], label: str) -> Enum:
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise CapabilityError(f"unknown-{label}")
    # Enum(value) formats the entire unknown value into a ValueError. Compare
    # against the fixed vocabulary without retaining it in exception context.
    for member in enum_type:
        if value == member.value:
            return member
    raise CapabilityError(f"unknown-{label}")


def _records(values: Iterable, record_type: type, maximum: int, label: str) -> tuple:
    if isinstance(values, (str, bytes, Mapping)):
        raise CapabilityError(f"invalid-{label}")
    try:
        owned = tuple(islice(iter(values), maximum + 1))
    except TypeError as exc:
        raise CapabilityError(f"invalid-{label}") from exc
    if len(owned) > maximum:
        raise CapabilityError(f"too-many-{label}")
    if any(type(value) is not record_type for value in owned):
        raise CapabilityError(f"invalid-{label}-record")
    return owned


@dataclass(frozen=True, slots=True)
class NamedLimit:
    """A declared upper capacity, or a requested minimum capacity in a requirement.

The operator schema defines the name and unit. Matching compares quantities
only for the same name and unit; it cannot infer joint shape/layout constraints.
    """

    name: str
    value: int
    unit: str = "count"

    def __post_init__(self) -> None:
        _identifier(self.name, "limit-name")
        _quantity(self.value, "limit-value")
        _identifier(self.unit, "limit-unit")


def _limits(values: Iterable[NamedLimit]) -> tuple[NamedLimit, ...]:
    owned = _records(values, NamedLimit, MAX_LIMITS, "limits")
    if len({value.name for value in owned}) != len(owned):
        raise CapabilityError("duplicate-limit-name")
    return owned


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Opaque device identity scoped to a backend, with optional declared labels."""

    backend_id: str
    device_id: str
    name: str
    vendor: str | None = None
    architecture: str | None = None
    driver_version: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.backend_id, "backend-id")
        _text(self.device_id, "device-id")
        _text(self.name, "device-name")
        for field in ("vendor", "architecture", "driver_version"):
            value = getattr(self, field)
            if value is not None:
                _text(value, field.replace("_", "-"))


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    """One exact declared operator/schema/payload-dtype/arithmetic combination."""

    name: str
    schema_version: int
    dtype: DType
    precision: Precision
    limits: tuple[NamedLimit, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "operator-name")
        _version(self.schema_version, "operator-schema-version")
        object.__setattr__(self, "dtype", _enum(self.dtype, DType, "dtype"))
        object.__setattr__(self, "precision", _enum(self.precision, Precision, "precision"))
        object.__setattr__(self, "limits", _limits(self.limits))

    @property
    def key(self) -> tuple[str, int, DType, Precision]:
        return self.name, self.schema_version, self.dtype, self.precision


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """A caller-declared result and optional reference, never verified on import.

Even PASSED evidence is not consumed by matching and is not training admission.
The digest identifies referenced bytes; no authentication or freshness is implied.
    """

    scope: str
    status: EvidenceStatus
    artifact_sha256: str | None = None
    artifact_uri: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.scope, "evidence-scope")
        object.__setattr__(self, "status", _enum(self.status, EvidenceStatus, "evidence-status"))
        if self.artifact_sha256 is not None:
            if (type(self.artifact_sha256) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256)):
                raise CapabilityError("invalid-evidence-sha256")
        if self.artifact_uri is not None:
            _text(self.artifact_uri, "evidence-uri", maximum=2048)


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    identity: DeviceIdentity
    operators: tuple[OperatorCapability, ...] = ()
    evidence: tuple[ValidationEvidence, ...] = ()
    schema_version: int = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.identity) is not DeviceIdentity:
            raise CapabilityError("invalid-device-identity")
        _version(self.schema_version, "capability-schema-version")
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise CapabilityError("unsupported-capability-schema-version")
        operators = _records(self.operators, OperatorCapability, MAX_OPERATORS, "operators")
        if len({value.key for value in operators}) != len(operators):
            raise CapabilityError("duplicate-operator-variant")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "evidence", _records(self.evidence, ValidationEvidence, MAX_EVIDENCE, "evidence"))


@dataclass(frozen=True, slots=True)
class OperatorRequirement:
    """Exact declared variant and lower bounds on named upper capacities."""

    name: str
    schema_version: int
    dtype: DType
    precision: Precision
    min_limits: tuple[NamedLimit, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "operator-name")
        _version(self.schema_version, "operator-schema-version")
        object.__setattr__(self, "dtype", _enum(self.dtype, DType, "dtype"))
        object.__setattr__(self, "precision", _enum(self.precision, Precision, "precision"))
        object.__setattr__(self, "min_limits", _limits(self.min_limits))

    @property
    def key(self) -> tuple[str, int, DType, Precision]:
        return self.name, self.schema_version, self.dtype, self.precision


def _requirements(values: Iterable[OperatorRequirement]) -> tuple[OperatorRequirement, ...]:
    owned = _records(values, OperatorRequirement, MAX_OPERATORS, "requirements")
    if not owned:
        raise CapabilityError("empty-requirements")
    if len({value.key for value in owned}) != len(owned):
        raise CapabilityError("duplicate-operator-requirement")
    return owned


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    """A declared-capability comparison; ``supported`` never means training ready."""

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reasons = _records(self.reasons, str, MAX_OPERATORS * (MAX_LIMITS + 1), "reasons")
        for reason in reasons:
            _text(reason, "reason", maximum=256)
        object.__setattr__(self, "reasons", reasons)

    @property
    def supported(self) -> bool:
        return not self.reasons


def match_capabilities(device: DeviceCapabilities,
                       requirements: Iterable[OperatorRequirement]) -> CapabilityMatch:
    """Compare declarations only, without probing, executing or admitting training.

All requirements must match the same device. Independent maximum capacities
cannot express coupled constraints; the future execution boundary must check
complete input shapes, layouts and values against each operator schema.
    """
    if type(device) is not DeviceCapabilities:
        raise CapabilityError("invalid-device-capabilities")
    requirements = _requirements(requirements)
    reasons = []
    for required in requirements:
        variants = [value for value in device.operators if value.name == required.name]
        if not variants:
            reasons.append(f"operator-unavailable:{required.name}")
            continue
        variants = [value for value in variants if value.schema_version == required.schema_version]
        if not variants:
            reasons.append(f"operator-schema-unsupported:{required.name}")
            continue
        variants = [value for value in variants if value.dtype == required.dtype]
        if not variants:
            reasons.append(f"operator-dtype-unsupported:{required.name}")
            continue
        variants = [value for value in variants if value.precision == required.precision]
        if not variants:
            reasons.append(f"operator-precision-unsupported:{required.name}")
            continue
        declared = {limit.name: limit for limit in variants[0].limits}
        for requested in required.min_limits:
            offered = declared.get(requested.name)
            if offered is None:
                reasons.append(f"limit-unavailable:{required.name}:{requested.name}")
            elif offered.unit != requested.unit:
                reasons.append(f"limit-unit-mismatch:{required.name}:{requested.name}")
            elif offered.value < requested.value:
                reasons.append(f"limit-insufficient:{required.name}:{requested.name}")
    return CapabilityMatch(tuple(reasons))
