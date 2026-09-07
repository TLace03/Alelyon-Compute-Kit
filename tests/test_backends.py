"""Registry tests use synthetic declarations, never hardware/provider objects."""

import builtins
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import FrozenInstanceError
import importlib
import importlib.metadata
from itertools import repeat
import socket
import subprocess
import sys

import pytest

from alelyon_compute_kit.backends import (
    BACKEND_API_VERSION, MAX_BACKENDS, BackendDescriptor, BackendRegistry,
    DeviceRejection, RegistryError, SelectionResult,
)
from alelyon_compute_kit.capabilities import (
    CapabilityError, DType, DeviceCapabilities, DeviceIdentity, OperatorCapability,
    OperatorRequirement, Precision,
)


def device(backend="test", device_id="synthetic:0", operators=("matmul",), vendor=None):
    return DeviceCapabilities(DeviceIdentity(backend, device_id, "Synthetic", vendor=vendor),
                              [OperatorCapability(name, 1, DType.FLOAT32, Precision.FP32) for name in operators])


def required(name="matmul"):
    return OperatorRequirement(name, 1, DType.FLOAT32, Precision.FP32)


def backend(backend_id="test", *, devices=None, api_version=BACKEND_API_VERSION):
    return BackendDescriptor(backend_id, "Synthetic backend", api_version,
                             [device(backend_id)] if devices is None else devices)


def test_registry_is_empty_until_explicit_registration():
    registry = BackendRegistry()
    assert registry.registered() == ()
    result = registry.select([required()])
    assert result.matches == () and result.reasons == ("no-backends-registered",)
    declaration = backend()
    assert registry.register(declaration) is None
    assert registry.registered() == (declaration,)
    assert registry.select([required()]).matches == declaration.devices
    assert BackendRegistry().registered() == ()


def test_duplicate_id_and_unsupported_api_refuse_without_mutating_registry():
    registry = BackendRegistry()
    original = backend()
    registry.register(original)
    with pytest.raises(RegistryError, match="duplicate-backend-id"):
        registry.register(backend(devices=[]))
    with pytest.raises(RegistryError, match="unsupported-backend-api-version"):
        registry.register(backend("future", api_version=BACKEND_API_VERSION + 1))
    assert registry.registered() == (original,)


@pytest.mark.parametrize("version", [True, False, 0, -1, 1.0, "1", 1 << 31])
def test_invalid_protocol_version_never_reaches_registration(version):
    with pytest.raises(CapabilityError, match="invalid-backend-api-version"):
        backend(api_version=version)


def test_registry_accepts_only_records_without_reading_provider_properties():
    calls = []
    class Provider:
        @property
        def backend_id(self):
            calls.append("property")
            raise AssertionError("provider property must not be read")
        def probe(self):
            calls.append("probe")
            raise AssertionError("probe must not run")
    registry = BackendRegistry()
    with pytest.raises(RegistryError, match="invalid-backend-descriptor"):
        registry.register(Provider())
    assert not calls and not registry.registered()


def test_explicit_backend_or_device_never_falls_back():
    registry = BackendRegistry()
    registry.register(backend("supported"))
    registry.register(backend("unsupported", devices=[device("unsupported", operators=())]))
    missing = registry.select([required()], backend_id="missing")
    assert missing.matches == () and missing.reasons == ("backend-not-registered",)
    unsupported = registry.select([required()], backend_id="unsupported")
    assert unsupported.matches == () and unsupported.rejected[0].reasons == ("operator-unavailable:matmul",)
    wrong_device = registry.select([required()], backend_id="supported", device_id="missing")
    assert wrong_device.matches == () and wrong_device.reasons == ("device-not-declared",)
    with pytest.raises(RegistryError, match="device-filter-requires-backend"):
        registry.select([required()], device_id="synthetic:0")


def test_no_devices_is_a_named_declaration_refusal():
    registry = BackendRegistry()
    registry.register(backend(devices=[]))
    assert registry.select([required()]).reasons == ("no-devices-declared",)


def test_selection_does_not_join_operators_across_devices():
    registry = BackendRegistry()
    registry.register(backend(devices=[device(device_id="a"), device(device_id="b", operators=("embedding",))]))
    result = registry.select([required(), required("embedding")])
    assert result.matches == () and result.reasons == ("no-declared-device-matches",)
    assert [value.reasons for value in result.rejected] == [
        ("operator-unavailable:embedding",), ("operator-unavailable:matmul",),
    ]


def test_all_matches_return_in_registration_order_without_vendor_ranking():
    registry = BackendRegistry()
    second = device("second", vendor="Vendor Z")
    first = device("first", vendor="Vendor A")
    registry.register(backend("second", devices=[second]))
    registry.register(backend("first", devices=[first]))
    result = registry.select([required()])
    assert result.matches == (second, first) and result.reasons == ()


def test_device_identifiers_are_scoped_to_backend():
    registry = BackendRegistry()
    registry.register(backend("a"))
    registry.register(backend("b"))
    result = registry.select([required()], backend_id="b", device_id="synthetic:0")
    assert len(result.matches) == 1 and result.matches[0].identity.backend_id == "b"


def test_backend_rejects_foreign_and_duplicate_devices():
    with pytest.raises(CapabilityError, match="device-backend-mismatch"):
        backend(devices=[device("different")])
    with pytest.raises(CapabilityError, match="duplicate-device-id"):
        backend(devices=[device(), device()])
    with pytest.raises(CapabilityError, match="too-many-devices"):
        backend(devices=repeat(device()))


def test_all_backend_result_collections_are_defensively_owned():
    devices = [device()]
    declaration = backend(devices=devices)
    registry = BackendRegistry()
    registry.register(declaration)
    registered = registry.registered()
    result = registry.select([required()])
    devices.clear()
    assert len(result.matches) == len(registered[0].devices) == 1
    reasons = ["operator-unavailable:embedding"]
    rejected = DeviceRejection(device(device_id="synthetic:1").identity, reasons)
    matches = [device()]
    rejections = [rejected]
    manual_result = SelectionResult(matches, rejections)
    refused_result = SelectionResult(reasons=reasons)
    reasons.clear()
    matches.clear()
    rejections.clear()
    assert len(manual_result.matches) == len(manual_result.rejected) == len(refused_result.reasons) == 1
    assert rejected.reasons == ("operator-unavailable:embedding",)
    with pytest.raises(FrozenInstanceError):
        declaration.devices = ()
    with pytest.raises(FrozenInstanceError):
        result.matches = ()


def test_registry_capacity_refuses_without_displacing_existing_entries():
    registry = BackendRegistry()
    for index in range(MAX_BACKENDS):
        registry.register(backend(f"b{index}", devices=[]))
    original = registry.registered()
    with pytest.raises(RegistryError, match="backend-registry-full"):
        registry.register(backend("extra", devices=[]))
    assert registry.registered() == original


def test_same_id_concurrent_registration_has_one_winner():
    registry = BackendRegistry()
    def register(_):
        try:
            registry.register(backend())
            return "registered"
        except RegistryError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(register, range(8)))
    assert outcomes.count("registered") == 1
    assert outcomes.count("duplicate-backend-id") == 7
    assert len(registry.registered()) == 1


@pytest.mark.parametrize("requirements", [[], None, "matmul", [object()]])
def test_malformed_requirements_refuse_before_empty_registry_selection(requirements):
    with pytest.raises(CapabilityError):
        BackendRegistry().select(requirements)


def test_import_and_registry_use_do_not_discover_plugins_probe_or_open_devices(monkeypatch):
    calls = []
    def forbidden(label):
        def fail(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"implicit side effect: {label}")
        return fail
    monkeypatch.setattr(importlib.metadata, "entry_points", forbidden("entry-points"))
    monkeypatch.setattr(ctypes, "CDLL", forbidden("native-library"))
    monkeypatch.setattr(subprocess, "Popen", forbidden("process"))
    monkeypatch.setattr(socket, "socket", forbidden("socket"))
    monkeypatch.setattr(socket, "create_connection", forbidden("connection"))
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "cupy", "vulkan", "pyopencl", "jax"}:
            return forbidden(f"accelerator-import:{name}")()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    saved = {name: module for name, module in sys.modules.items()
             if name == "alelyon_compute_kit" or name.startswith("alelyon_compute_kit.")}
    for name in saved:
        del sys.modules[name]
    try:
        module = importlib.import_module("alelyon_compute_kit.backends")
        caps = importlib.import_module("alelyon_compute_kit.capabilities")
        registry = module.BackendRegistry()
        registry.register(module.BackendDescriptor("fake", "Explicit empty declaration"))
        result = registry.select([caps.OperatorRequirement("matmul", 1, "float32", "fp32")])
        assert result.reasons == ("no-devices-declared",)
        assert calls == []
    finally:
        for name in tuple(sys.modules):
            if name == "alelyon_compute_kit" or name.startswith("alelyon_compute_kit."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_mutable_backend_identifier_cannot_bypass_duplicate_registration():
    calls = []
    class MutableIdentifier(str):
        def __hash__(self):
            calls.append("hash")
            return self.hash_value
    backend_id = MutableIdentifier("synthetic")
    backend_id.hash_value = 0
    registry = BackendRegistry()
    with pytest.raises(CapabilityError, match="invalid-backend-id"):
        declaration = BackendDescriptor(backend_id, "Synthetic")
        registry.register(declaration)
        backend_id.hash_value = 1
        registry.register(declaration)
    assert registry.registered() == ()
    assert calls == []


@pytest.mark.parametrize("field,value", [("backend_id", "test"), ("name", "Synthetic")])
def test_backend_labels_refuse_string_subclasses(field, value):
    class StringSubclass(str):
        pass
    values = dict(backend_id="test", name="Synthetic")
    values[field] = StringSubclass(value)
    with pytest.raises(CapabilityError):
        BackendDescriptor(**values)


@pytest.mark.parametrize("field,value", [("backend_id", "test"), ("device_id", "synthetic:0")])
def test_selection_filters_refuse_string_subclasses(field, value):
    class StringSubclass(str):
        pass
    registry = BackendRegistry()
    registry.register(backend())
    filters = dict(backend_id="test")
    filters[field] = StringSubclass(value)
    with pytest.raises(CapabilityError):
        registry.select([required()], **filters)


@pytest.mark.parametrize("case,reason", [
    ("duplicate-matches", "duplicate-matched-device"),
    ("duplicate-rejections", "duplicate-rejected-device"),
    ("matched-and-rejected", "device-both-matched-and-rejected"),
    ("matches-and-refusal", "matches-with-global-refusal"),
])
def test_contradictory_selection_records_refuse(case, reason):
    one = device()
    same_identity_other_metadata = device(vendor="Different label")
    rejection = DeviceRejection(one.identity, ["operator-unavailable:embedding"])
    other_rejection = DeviceRejection(same_identity_other_metadata.identity, ["operator-unavailable:mm"])
    fields = {
        "duplicate-matches": dict(matches=[one, same_identity_other_metadata]),
        "duplicate-rejections": dict(rejected=[rejection, other_rejection]),
        "matched-and-rejected": dict(matches=[one], rejected=[other_rejection]),
        "matches-and-refusal": dict(matches=[one], reasons=["no-backends-registered"]),
    }
    with pytest.raises(CapabilityError, match=reason):
        SelectionResult(**fields[case])


def test_selection_identity_consistency_is_backend_scoped():
    one = device("one")
    two = device("two")
    valid = SelectionResult(matches=[one], rejected=[DeviceRejection(two.identity, ["unsupported"])])
    assert valid.matches == (one,) and valid.rejected[0].identity == two.identity
