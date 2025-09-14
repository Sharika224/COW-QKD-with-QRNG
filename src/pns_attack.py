"""
pns_attack.py
A stylized Photon-Number Splitting (PNS) attack model for weak coherent pulses.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class PNSConfig:
    tap_efficiency: float = 0.5   # Eve's beamsplitter effectiveness (0..1)
    leak_prob: float = 0.9        # Probability Eve succeeds when photons>1
    min_photons_to_split: int = 2 # Eve can split only multiphoton pulses

def apply_pns(photon_counts: np.ndarray, cfg: PNSConfig) -> np.ndarray:
    """
    Return a boolean array 'compromised' the same shape as photon_counts.
    True means Eve obtained information on that symbol via PNS.
    """
    compromised = np.zeros_like(photon_counts, dtype=np.uint8)
    multi = photon_counts >= cfg.min_photons_to_split
    n_multi = int(np.count_nonzero(multi))
    if n_multi == 0:
        return compromised

    # Eve attempts on multiphoton pulses; success depends on tap_efficiency & leak_prob
    success_prob = np.clip(cfg.tap_efficiency * cfg.leak_prob, 0.0, 1.0)
    compromised[multi] = (np.random.rand(n_multi) < success_prob).astype(np.uint8)
    return compromised
