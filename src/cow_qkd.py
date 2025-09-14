"""
cow_qkd.py
COW-QKD simulator with QRNG and optional PNS attack.
Run as a module from the repo root:  python -m src.cow_qkd --keys 10000 --attack pns
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# package-relative imports (require running as a module: python -m src.cow_qkd)
from qrng import set_seed, qrng_bits, monobit_test
from pns_attack import apply_pns, PNSConfig
from analysis import summarize_and_save
from visualization import plot_histogram, plot_compromise_rates


# ----------------- config -----------------
@dataclass
class ChannelConfig:
    mu: float = 0.2               # mean photons per pulse
    channel_loss: float = 0.2     # 0..1
    dark_click_prob: float = 1e-3
    decoy_prob: float = 0.15
    detector_efficiency: float = 0.8


# ----------------- sim helpers -----------------
def generate_photon_counts(n: int, mu: float, channel_loss: float) -> np.ndarray:
    src = np.random.poisson(mu, size=n)
    survive = np.random.binomial(src, p=max(0.0, 1.0 - channel_loss))
    return survive

def simulate_detection(photon_counts: np.ndarray, dark_click_prob: float, detector_efficiency: float) -> np.ndarray:
    clicks = np.zeros_like(photon_counts, dtype=np.uint8)
    for i, k in enumerate(photon_counts):
        detected = False
        if k > 0:
            p_detect = 1.0 - (1.0 - detector_efficiency) ** int(k)
            detected = (np.random.rand() < p_detect)
        if not detected and np.random.rand() < dark_click_prob:
            detected = True
        clicks[i] = 1 if detected else 0
    return clicks

def sift_key(tx_bits: np.ndarray, clicks: np.ndarray, decoy_flags: np.ndarray) -> np.ndarray:
    keep = (decoy_flags == 0) & (clicks == 1)
    return tx_bits[keep]

def baseline_pns_probability(photon_counts: np.ndarray) -> float:
    multi = photon_counts >= 2
    if multi.sum() == 0:
        return 0.0
    return 0.8 * (multi.mean())


# ----------------- main -----------------
def main() -> int:
    parser = argparse.ArgumentParser(description="COW-QKD + QRNG simulator with optional PNS attack")
    parser.add_argument("--keys", type=int, default=10000)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--channel-loss", type=float, default=0.2)
    parser.add_argument("--dark", type=float, default=1e-3)
    parser.add_argument("--decoy", type=float, default=0.15)
    parser.add_argument("--det-eff", type=float, default=0.8)
    parser.add_argument("--attack", type=str, default="none", choices=["none", "pns"])
    parser.add_argument("--tap", type=float, default=0.5)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-prefix", type=str, default="run")
    args = parser.parse_args()

    set_seed(args.seed)

    n = int(args.keys)
    ch_cfg = ChannelConfig(
        mu=args.mu, channel_loss=args.channel_loss,
        dark_click_prob=args.dark, decoy_prob=args.decoy,
        detector_efficiency=args.det_eff
    )

    # 1) QRNG bits
    tx_bits = qrng_bits(n)
    sanity = monobit_test(tx_bits)

    # 2) decoys
    decoy_flags = (np.random.rand(n) < ch_cfg.decoy_prob).astype(np.uint8)

    # 3) channel
    photon_counts = generate_photon_counts(n, ch_cfg.mu, ch_cfg.channel_loss)

    # 4) attack
    compromised = np.zeros(n, dtype=np.uint8)
    baseline_prob = baseline_pns_probability(photon_counts)
    if args.attack == "pns":
        cfg = PNSConfig(tap_efficiency=args.tap, leak_prob=args.leak, min_photons_to_split=2)
        compromised = apply_pns(photon_counts, cfg)

    # 5) detection
    clicks = simulate_detection(photon_counts, ch_cfg.dark_click_prob, ch_cfg.detector_efficiency)

    # 6) sifting
    rx_bits = sift_key(tx_bits, clicks, decoy_flags)

    # 7) outputs (always relative to repo root)
    repo_root   = Path(__file__).resolve().parents[1]
    data_results = repo_root / "data" / "results"
    data_keys    = repo_root / "data" / "keys"
    results_dir  = repo_root / "results"
    data_results.mkdir(parents=True, exist_ok=True)
    data_keys.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_results / f"{args.out_prefix}_symbols.csv"
    metrics = summarize_and_save(
        run_id=args.out_prefix,
        tx_bits=tx_bits,
        rx_bits=rx_bits,
        compromised_flags=compromised,
        baseline_attack_prob=baseline_prob,
        out_csv_path=str(csv_path),
    )

    # 8) plots
    plot_histogram(photon_counts, "Photon Counts After Channel", "photons", str(results_dir / "photon_counts.png"))
    before = (np.random.rand(compromised.size) < baseline_prob).astype(np.uint8)
    plot_compromise_rates(before, compromised, str(results_dir / "pns_reduction.png"))

    # 9) save sifted key
    key_path = data_keys / f"{args.out_prefix}_sifted_key.bin"
    rx_bits.astype(np.uint8).tofile(str(key_path))

    # 10) summary
    print("=== COW-QKD Simulation Summary ===")
    print(f"Requested TX bits      : {metrics.requested_keys}")
    print(f"Sifted key bits (RX)   : {metrics.sifted_keys}")
    print(f"QBER                   : {metrics.qber:.4f}")
    print(f"Attack success prob    : {metrics.attack_success_prob:.4f}")
    if metrics.reduction_vs_baseline is not None:
        print(f"Reduction vs baseline  : {metrics.reduction_vs_baseline*100:.1f}%")
    print(f"QRNG sanity            : n={sanity['n']} ones={sanity['ones']} zeros={sanity['zeros']} balance={sanity['balance']:.4f}")
    print(f"Outputs: {csv_path}, {key_path}, {results_dir}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
