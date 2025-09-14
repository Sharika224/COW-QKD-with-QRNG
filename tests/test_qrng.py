import numpy as np
from qrng import set_seed, qrng_bits, monobit_test

def test_qrng_bits_shape_and_values():
    set_seed(0)
    bits = qrng_bits(256)
    assert bits.shape == (256,)
    uniq = set(np.unique(bits))
    assert uniq.issubset({0, 1})

def test_qrng_reproducibility_with_seed():
    set_seed(123)
    a = qrng_bits(128).copy()
    set_seed(123)
    b = qrng_bits(128).copy()
    assert np.array_equal(a, b)

def test_monobit_balance_reasonable():
    set_seed(999)
    bits = qrng_bits(5000)
    stats = monobit_test(bits)
    # very loose sanity bound (should pass consistently)
    assert 0 <= stats["balance"] < 0.1
