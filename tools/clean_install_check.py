"""Run with the freshly installed interpreter, outside the source checkout.

Only the final explicit flag loads the DLL to query ABI/schema; it never opens
a device. The default check proves root imports cannot load native/optional ML.
"""
from __future__ import annotations

import argparse
import builtins
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-identity", action="store_true")
    args = parser.parse_args()
    original = builtins.__import__

    def guarded(name, *pos, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "ctypes", "subprocess"}:
            raise AssertionError("root import attempted optional/native module")
        return original(name, *pos, **kwargs)

    builtins.__import__ = guarded
    try:
        import alelyon_compute_kit as package
    finally:
        builtins.__import__ = original
    assert package.__version__ == "0.1.0a3"
    assert "site-packages" in pathlib.Path(package.__file__).parts
    assert package.BackendRegistry().registered() == ()
    if args.native_identity:
        import ctypes
        from alelyon_compute_kit import ack
        path = ack.library_path()
        assert path is not None and path.is_relative_to(pathlib.Path(package.__file__).parent)
        library = ctypes.CDLL(str(path))
        library.ack_vq_schema.argtypes = []
        library.ack_vq_schema.restype = ctypes.c_uint32
        assert library.ack_vq_schema() == 1
        assert ack.abi_info().version == 10
    print("PASS: installed import/identity boundary; device execution UNMEASURED")


if __name__ == "__main__":
    main()
