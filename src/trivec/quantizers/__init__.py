"""Quantizers: one file per compression method, all implementing `Quantizer`."""

from collections.abc import Callable
from typing import Any

from trivec.quantizers.base import Codes, FloatArray, IntArray, Quantizer
from trivec.quantizers.binary import BinaryQuantizer
from trivec.quantizers.float32 import Float32Quantizer
from trivec.quantizers.int8 import Int8Quantizer
from trivec.quantizers.ternary import TernaryQuantizer

REGISTRY: dict[str, Callable[..., Quantizer]] = {
    "float32": Float32Quantizer,
    "int8": Int8Quantizer,
    "binary": BinaryQuantizer,
    "ternary": TernaryQuantizer,
}


def make_quantizer(name: str, **params: Any) -> Quantizer:
    """Build a quantizer from its config name and parameters (e.g. from YAML)."""
    if name == "pq":  # imported lazily: faiss is heavy and optional for tests
        from trivec.quantizers.pq import PQQuantizer

        return PQQuantizer(**params)
    try:
        factory = REGISTRY[name]
    except KeyError:
        known = sorted([*REGISTRY, "pq"])
        raise ValueError(f"unknown quantizer {name!r}; known: {known}") from None
    return factory(**params)


__all__ = [
    "REGISTRY",
    "BinaryQuantizer",
    "Codes",
    "Float32Quantizer",
    "FloatArray",
    "Int8Quantizer",
    "IntArray",
    "Quantizer",
    "TernaryQuantizer",
    "make_quantizer",
]
