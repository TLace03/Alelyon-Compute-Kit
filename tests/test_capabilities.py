"""Pure declaration tests. No hardware support is registered or exercised."""

from dataclasses import FrozenInstanceError, replace
from enum import Enum
from itertools import repeat
import traceback

import pytest

from alelyon_compute_kit.capabilities import (
    CAPABILITY_SCHEMA_VERSION, MAX_EVIDENCE, MAX_LIMITS, MAX_OPERATORS, MAX_QUANTITY,
    CapabilityError, CapabilityMatch, DType, DeviceCapabilities, DeviceIdentity,
    EvidenceStatus, NamedLimit, OperatorCapability, OperatorRequirement, Precision,
    ValidationEvidence, match_capabilities,
)


def identity(**changes):
    values = dict(backend_id="test", device_id="synthetic:0", name="Synthetic declaration")
    values.update(changes)
    return DeviceIdentity(**values)


def operator(**changes):
    values = dict(name="matmul", schema_version=1, dtype=DType.FLOAT32, precision=Precision.FP32,
                  limits=(NamedLimit("elements", 1024, "elements"),))
    values.update(changes)
    return OperatorCapability(**values)


def requirement(**changes):
    values = dict(name="matmul", schema_version=1, dtype=DType.FLOAT32, precision=Precision.FP32,
                  min_limits=(NamedLimit("elements", 1024, "elements"),))
    values.update(changes)
    return OperatorRequirement(**values)


def test_exact_declared_match_and_limit_boundary():
    device = DeviceCapabilities(identity(), [operator()])
    assert match_capabilities(device, [requirement()]).supported
    result = match_capabilities(device, [requirement(min_limits=[NamedLimit("elements", 1025, "elements")])])
    assert result.reasons == ("limit-insufficient:matmul:elements",)
    assert not result.supported


@pytest.mark.parametrize(("changes", "reason"), [
    ({"name": "embedding"}, "operator-unavailable:embedding"),
    ({"schema_version": 2}, "operator-schema-unsupported:matmul"),
    ({"dtype": DType.FLOAT16}, "operator-dtype-unsupported:matmul"),
    ({"precision": Precision.BF16_FP32}, "operator-precision-unsupported:matmul"),
    ({"min_limits": [NamedLimit("rows", 1)]}, "limit-unavailable:matmul:rows"),
    ({"min_limits": [NamedLimit("elements", 1, "bytes")]}, "limit-unit-mismatch:matmul:elements"),
])
def test_named_unsupported_reasons(changes, reason):
    result = match_capabilities(DeviceCapabilities(identity(), [operator()]), [requirement(**changes)])
    assert result.reasons == (reason,)


def test_precision_is_not_cartesian_product_of_independent_dtype_lists():
    device = DeviceCapabilities(identity(), [
        operator(dtype=DType.FLOAT16, precision=Precision.FP16_FP32),
        operator(dtype=DType.FLOAT32, precision=Precision.FP32),
    ])
    result = match_capabilities(device, [requirement(dtype=DType.FLOAT32, precision=Precision.FP16_FP32)])
    assert result.reasons == ("operator-precision-unsupported:matmul",)


def test_all_requirements_are_checked_and_refusals_are_complete():
    device = DeviceCapabilities(identity(), [operator()])
    requested = [requirement(min_limits=[NamedLimit("rows", 1), NamedLimit("elements", 2048, "elements")]),
                 requirement(name="embedding")]
    assert match_capabilities(device, requested).reasons == (
        "limit-unavailable:matmul:rows", "limit-insufficient:matmul:elements", "operator-unavailable:embedding",
    )


@pytest.mark.parametrize("status", list(EvidenceStatus))
def test_evidence_and_vendor_do_not_admit_training_or_change_matching(status):
    bare = DeviceCapabilities(identity(), [operator()])
    decorated = DeviceCapabilities(identity(vendor="Any vendor", architecture="Any architecture"),
                                   [operator()], [ValidationEvidence("training", status, "a" * 64,
                                                                    "https://example.invalid/not-read")])
    assert match_capabilities(bare, [requirement()]) == match_capabilities(decorated, [requirement()])
    empty = replace(decorated, operators=())
    assert match_capabilities(empty, [requirement()]).reasons == ("operator-unavailable:matmul",)
    assert not hasattr(decorated, "training_ready")


def test_constructor_defensively_owns_all_collections():
    limits = [NamedLimit("elements", 1024, "elements")]
    op = operator(limits=limits)
    needed = requirement(min_limits=limits)
    operators = [op]
    evidence = [ValidationEvidence("operator.matmul", EvidenceStatus.UNMEASURED)]
    device = DeviceCapabilities(identity(), operators, evidence)
    reasons = ["operator-unavailable:embedding"]
    match = CapabilityMatch(reasons)
    limits.clear()
    operators.clear()
    evidence.clear()
    reasons.clear()
    assert len(op.limits) == len(needed.min_limits) == len(device.operators) == len(device.evidence) == 1
    assert not match.supported
    assert match_capabilities(device, [needed]).supported
    assert isinstance(device.operators, tuple)


@pytest.mark.parametrize("record,field,new_value", [
    (identity(), "name", "changed"),
    (operator(), "precision", Precision.BF16_FP32),
    (requirement(), "schema_version", 2),
    (NamedLimit("n", 1), "value", 2),
    (ValidationEvidence("training", "unmeasured"), "status", EvidenceStatus.PASSED),
    (DeviceCapabilities(identity()), "operators", (operator(),)),
    (CapabilityMatch(), "reasons", ("changed",)),
])
def test_records_are_frozen(record, field, new_value):
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, new_value)


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None, MAX_QUANTITY + 1])
def test_limit_quantity_refuses_coercion_and_overflow(value):
    with pytest.raises(CapabilityError, match="invalid-limit-value"):
        NamedLimit("elements", value)


def test_quantity_endpoints_remain_exact():
    assert NamedLimit("zero", 0).value == 0
    assert NamedLimit("large", MAX_QUANTITY).value == MAX_QUANTITY
    device = DeviceCapabilities(identity(), [operator(limits=[NamedLimit("large", MAX_QUANTITY)])])
    assert match_capabilities(device, [requirement(min_limits=[NamedLimit("large", MAX_QUANTITY)])]).supported


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1", 1 << 31])
@pytest.mark.parametrize("factory", [operator, requirement])
def test_operator_versions_refuse_invalid_values(factory, value):
    with pytest.raises(CapabilityError, match="invalid-operator-schema-version"):
        factory(schema_version=value)


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_capability_versions_refuse_invalid_values(value):
    with pytest.raises(CapabilityError, match="invalid-capability-schema-version"):
        DeviceCapabilities(identity(), schema_version=value)


def test_unknown_capability_schema_refuses():
    with pytest.raises(CapabilityError, match="unsupported-capability-schema-version"):
        DeviceCapabilities(identity(), schema_version=CAPABILITY_SCHEMA_VERSION + 1)


@pytest.mark.parametrize("field,value,reason", [
    ("dtype", "float8", "unknown-dtype"),
    ("dtype", True, "unknown-dtype"),
    ("precision", "automatic", "unknown-precision"),
    ("precision", True, "unknown-precision"),
    ("precision", None, "unknown-precision"),
])
@pytest.mark.parametrize("factory", [operator, requirement])
def test_unknown_dtype_or_precision_refuses(factory, field, value, reason):
    with pytest.raises(CapabilityError, match=reason):
        factory(**{field: value})


def test_known_string_enum_values_are_canonicalized():
    op = operator(dtype="float32", precision="fp32")
    assert op.dtype is DType.FLOAT32 and op.precision is Precision.FP32
    assert ValidationEvidence("training", "unmeasured").status is EvidenceStatus.UNMEASURED


@pytest.mark.parametrize("value", ["", " invalid", "invalid ", "a\nline", "x" * 129, True])
def test_device_labels_are_bounded(value):
    with pytest.raises(CapabilityError, match="invalid-device-name"):
        identity(name=value)


@pytest.mark.parametrize("value", ["", "Bad", "a/b", "a:b", "x" * 65, True])
def test_identifiers_are_bounded_closed_syntax(value):
    with pytest.raises(CapabilityError, match="invalid-operator-name"):
        operator(name=value)


def test_duplicate_variants_limits_and_requirements_refuse():
    with pytest.raises(CapabilityError, match="duplicate-operator-variant"):
        DeviceCapabilities(identity(), [operator(), operator()])
    with pytest.raises(CapabilityError, match="duplicate-limit-name"):
        operator(limits=[NamedLimit("size", 1), NamedLimit("size", 2, "bytes")])
    with pytest.raises(CapabilityError, match="duplicate-operator-requirement"):
        match_capabilities(DeviceCapabilities(identity()), [requirement(), requirement()])


@pytest.mark.parametrize("values", [None, "matmul", {"matmul": operator()}, [object()]])
def test_invalid_operator_collections_refuse(values):
    with pytest.raises(CapabilityError):
        DeviceCapabilities(identity(), values)


def test_collection_bounds_stop_infinite_input_iterators():
    with pytest.raises(CapabilityError, match="too-many-operators"):
        DeviceCapabilities(identity(), repeat(operator()))
    with pytest.raises(CapabilityError, match="too-many-limits"):
        operator(limits=repeat(NamedLimit("size", 1)))
    with pytest.raises(CapabilityError, match="too-many-evidence"):
        DeviceCapabilities(identity(), evidence=repeat(ValidationEvidence("x", "unmeasured")))
    assert MAX_OPERATORS == 4096 and MAX_LIMITS == 64 and MAX_EVIDENCE == 128


@pytest.mark.parametrize("values", [[], (), None, "matmul", [object()]])
def test_empty_or_malformed_requirements_cannot_vacuously_match(values):
    with pytest.raises(CapabilityError):
        match_capabilities(DeviceCapabilities(identity()), values)


@pytest.mark.parametrize("changes,reason", [
    ({"status": "verified"}, "unknown-evidence-status"),
    ({"artifact_sha256": "a" * 63}, "invalid-evidence-sha256"),
    ({"artifact_sha256": "z" * 64}, "invalid-evidence-sha256"),
    ({"artifact_sha256": True}, "invalid-evidence-sha256"),
    ({"artifact_uri": "x" * 2049}, "invalid-evidence-uri"),
])
def test_malformed_evidence_refuses(changes, reason):
    fields = dict(scope="training", status=EvidenceStatus.UNMEASURED)
    fields.update(changes)
    with pytest.raises(CapabilityError, match=reason):
        ValidationEvidence(**fields)


@pytest.mark.parametrize("factory,value", [
    (lambda value: identity(backend_id=value), "test"),
    (lambda value: identity(device_id=value), "device-0"),
    (lambda value: identity(name=value), "Synthetic"),
    (lambda value: identity(vendor=value), "Vendor"),
    (lambda value: identity(architecture=value), "Architecture"),
    (lambda value: identity(driver_version=value), "1.0"),
    (lambda value: NamedLimit(value, 1), "elements"),
    (lambda value: NamedLimit("elements", 1, value), "count"),
    (lambda value: operator(name=value), "matmul"),
    (lambda value: requirement(name=value), "matmul"),
    (lambda value: ValidationEvidence(value, "unmeasured"), "training"),
    (lambda value: ValidationEvidence("training", "unmeasured", artifact_uri=value), "https://example.invalid"),
    (lambda value: ValidationEvidence("training", "unmeasured", artifact_sha256=value), "a" * 64),
])
def test_scalar_string_subclasses_are_not_retained_as_immutable_data(factory, value):
    class StringSubclass(str):
        pass
    with pytest.raises(CapabilityError):
        factory(StringSubclass(value))


@pytest.mark.parametrize("factory", [
    lambda value: operator(dtype=value),
    lambda value: operator(precision=value),
    lambda value: requirement(dtype=value),
    lambda value: requirement(precision=value),
    lambda value: ValidationEvidence("training", value),
])
def test_enum_equality_spoof_refuses_without_executing_comparison_callback(factory):
    calls = []
    class EqualToEverything:
        __hash__ = None
        def __eq__(self, other):
            calls.append("equality")
            return True
    with pytest.raises(CapabilityError, match="unknown-"):
        factory(EqualToEverything())
    assert calls == []


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("precision", "fp32")])
def test_other_string_enum_is_not_a_declared_enum(field, value):
    class OtherEnum(str, Enum):
        VALUE = value
    with pytest.raises(CapabilityError, match="unknown-"):
        operator(**{field: OtherEnum.VALUE})


@pytest.mark.parametrize("dtype", list(DType))
def test_declared_dtype_members_remain_supported(dtype):
    assert operator(dtype=dtype).dtype is dtype


@pytest.mark.parametrize("precision", list(Precision))
def test_declared_precision_members_remain_supported(precision):
    assert operator(precision=precision).precision is precision


@pytest.mark.parametrize("factory", [
    lambda value: operator(dtype=value),
    lambda value: operator(precision=value),
    lambda value: ValidationEvidence("training", value),
])
def test_unknown_enum_diagnostics_are_bounded_and_value_free(factory):
    value = "unrecognized-value-" * 10000
    with pytest.raises(CapabilityError, match="unknown-") as caught:
        factory(value)
    assert len(str(caught.value)) < 64
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert value not in "".join(traceback.format_exception(caught.value))
