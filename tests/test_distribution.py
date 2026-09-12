"""Host-only public source boundary checks; no DLL or device construction."""
from pathlib import Path
import importlib.util
import sys

import pytest


def backend():
    path = Path(__file__).resolve().parents[1] / "tools/_build_backend.py"
    spec = importlib.util.spec_from_file_location("_tested_backend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reviewed_source_manifest_and_version():
    module = backend()
    files = module.sources()
    assert b'__version__ = "0.1.0a2"' in files["src/alelyon_compute_kit/__init__.py"]
    assert module.DLL not in files
    assert module.RECEIPT not in files


def test_no_implicit_native_builder(tmp_path):
    module = backend()
    if not (module.ROOT / module.DLL).exists():
        with pytest.raises(module.PackageRefused, match="source-missing"):
            module.build_wheel(tmp_path)
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["../x", "C:/x", "a\\b", "/x", "a//b"])
def test_archive_names_refuse(path):
    module = backend()
    with pytest.raises(module.PackageRefused):
        module.safe_name(path)
