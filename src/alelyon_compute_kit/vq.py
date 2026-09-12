"""Optional NumPy VQ storage/dispatch API; no import-time device initialization."""
from ._kit.python.vq import VQAdamW, VQArray, VQDevice, encode, fit_codebook, storage_layout

__all__ = ["VQAdamW", "VQArray", "VQDevice", "encode", "fit_codebook", "storage_layout"]
