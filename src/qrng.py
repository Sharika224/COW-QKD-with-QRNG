"""
qrng.py
A lightweight quantum random number generator simulation (unbiased bit stream).
In practice this would be sourced from hardware; here we model ideal behavior.
"""
from __future__ import annotations
import numpy as np

def set_seed(seed: int | None = None) -> None:
    if seed is not None:
        np.random.seed(seed)

def qrng_bits(n_bits: int) -> np.ndarray:
    """Return n_bits of unbiased random bits (0/1)."""
    if n_bits <= 0:
        return np.zeros(0, dtype=np.uint8)
    # Using NumPy RNG as a stand-in for a QRNG. Replace with hardware API if available.
    return (np.random.rand(n_bits) < 0.5).astype(np.uint8)

def monobit_test(bits: np.ndarray) -> dict:
    """Simple sanity check: count of ones vs zeros."""
    if bits.size == 0:
        return {"n": 0, "ones": 0, "zeros": 0, "balance": 0.0}
    ones = int(bits.sum())
    zeros = int(bits.size - ones)
    balance = abs(ones - zeros) / bits.size
    return {"n": int(bits.size), "ones": ones, "zeros": zeros, "balance": balance}
