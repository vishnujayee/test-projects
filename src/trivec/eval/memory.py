"""How much memory a quantizer keeps OUTSIDE the per-vector codes.

Two uses:
- Honest accounting. PQ's "48 bytes per vector" leaves out its codebooks, which matter a lot
  on a small corpus. We report bytes_per_vector_amortized = codes + shared state / n.
- Cheat detection. A quantizer that memorises the corpus in `self` could claim 1 byte per
  vector and perfect recall. Its shared state grows with n, and the Lab checks for that.
"""

import sys
import types
from typing import Any

import numpy as np

_SKIP = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)


def deep_size(obj: Any, _seen: set[int] | None = None) -> int:
    """Bytes held by obj, following containers and instance attributes."""
    seen = _seen if _seen is not None else set()
    if id(obj) in seen or isinstance(obj, _SKIP):
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, bytes | bytearray | str):
        return len(obj)
    if isinstance(obj, dict):
        return sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    if isinstance(obj, list | tuple | set | frozenset):
        return sum(deep_size(v, seen) for v in obj)
    if isinstance(obj, bool | int | float | complex) or obj is None:
        return int(sys.getsizeof(obj)) if isinstance(obj, int) and obj.bit_length() > 64 else 8
    state = getattr(obj, "__dict__", None)
    return deep_size(state, seen) if isinstance(state, dict) else 0


def shared_state_bytes(quantizer: Any) -> int:
    """Memory the quantizer keeps in its own attributes (scales, codebooks, ...)."""
    return deep_size(quantizer)
