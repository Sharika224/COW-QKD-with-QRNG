import numpy as np
from pns_attack import apply_pns, PNSConfig

def test_pns_flags_shape_and_values():
    np.random.seed(0)
    counts = np.random.poisson(0.3, size=1000)
    flags = apply_pns(counts, PNSConfig())
    assert flags.shape == counts.shape
    assert set(np.unique(flags)).issubset({0, 1})

def test_pns_higher_multi_photon_increases_compromise_rate():
    np.random.seed(1)
    # Construct two scenarios with different multi-photon rates
    few_multi = np.array([0,1,1,2] * 250)     # ~25% multi-photon
    more_multi = np.array([2,2,3,0] * 250)    # ~75% multi-photon
    cfg = PNSConfig(tap_efficiency=0.5, leak_prob=0.9, min_photons_to_split=2)
    r1 = apply_pns(few_multi, cfg).mean()
    r2 = apply_pns(more_multi, cfg).mean()
    assert r2 >= r1
