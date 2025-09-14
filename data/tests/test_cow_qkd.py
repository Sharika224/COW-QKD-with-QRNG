import numpy as np
from cow_qkd import generate_photon_counts, simulate_detection, sift_key, baseline_pns_probability

def test_generate_photon_counts_nonnegative_and_shape():
    np.random.seed(0)
    n = 2000
    counts = generate_photon_counts(n, mu=0.2, channel_loss=0.2)
    assert counts.shape == (n,)
    assert (counts >= 0).all()

def test_simulate_detection_and_sifting_pipeline():
    np.random.seed(42)
    n = 1000
    tx = (np.random.rand(n) < 0.5).astype(np.uint8)
    counts = generate_photon_counts(n, mu=0.25, channel_loss=0.15)
    clicks = simulate_detection(counts, dark_click_prob=1e-3, detector_efficiency=0.8)
    decoys = (np.random.rand(n) < 0.1).astype(np.uint8)
    rx = sift_key(tx, clicks, decoys)

    # sifted key cannot be longer than tx; must be uint8
    assert rx.size <= tx.size
    assert rx.dtype == np.uint8

def test_baseline_pns_probability_reasonable():
    np.random.seed(7)
    counts = np.random.poisson(0.3, size=5000)
    p = baseline_pns_probability(counts)
    # should be in [0, 0.8], and >0 if there are any multi-photon pulses
    assert 0.0 <= p <= 0.8
    assert (counts >= 2).any() == (p > 0.0)
