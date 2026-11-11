"""Bit packing helpers shared by the binary and ternary quantizers.

A "bit-plane" is a matrix of 0/1 values, one per dimension. We pack it into bytes,
padding each row to a whole number of 64-bit words so the packed search can view
rows as uint64 and use popcount.
"""

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
ByteArray = npt.NDArray[np.uint8]


def plane_bytes(dim: int) -> int:
    """Bytes needed for one bit-plane of `dim` bits, padded to whole 64-bit words."""
    return ((dim + 63) // 64) * 8


def pack_plane(bits: BoolArray) -> ByteArray:
    """bool [n, d] -> uint8 [n, plane_bytes(d)], little-endian bit order."""
    n, d = bits.shape
    padded_bits = plane_bytes(d) * 8
    if padded_bits != d:
        pad = np.zeros((n, padded_bits - d), dtype=np.bool_)
        bits = np.concatenate([bits, pad], axis=1)
    return np.packbits(bits, axis=1, bitorder="little")


def unpack_plane(packed: ByteArray, dim: int) -> BoolArray:
    """uint8 [n, plane_bytes(dim)] -> bool [n, dim]."""
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    return bits[:, :dim].astype(np.bool_)


def popcount_rows(words: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    """Count 1-bits in each row of a uint64 matrix."""
    return np.bitwise_count(words).sum(axis=1, dtype=np.int64)
