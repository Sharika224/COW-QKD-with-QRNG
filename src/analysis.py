"""
analysis.py
Utilities to compute key metrics, save CSVs, and summarize outcomes.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    requested_keys: int
    sifted_keys: int
    qber: float
    attack_success_prob: float
    reduction_vs_baseline: float | None = None


def compute_qber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    QBER is computed only over positions that made it into the sifted key.
    Since rx_bits is a compacted (sifted) sequence, compare over min length.
    """
    n = min(tx_bits.size, rx_bits.size)
    if n == 0:
        return 0.0
    return float(np.count_nonzero(tx_bits[:n] != rx_bits[:n])) / n


def attack_success_probability(compromised_flags: np.ndarray) -> float:
    if compromised_flags.size == 0:
        return 0.0
    return float(np.mean(compromised_flags.astype(np.float32)))


def summarize_and_save(
    run_id: str,
    tx_bits: np.ndarray,
    rx_bits: np.ndarray,
    compromised_flags: np.ndarray,
    baseline_attack_prob: float | None,
    out_csv_path: str,
) -> Metrics:
    """
    Save *aligned* data only. rx_bits is sifted (shorter), so we DO NOT mix it
    into the symbolwise CSV. Instead:
      - CSV: tx (len N), compromised (len N)  -> equal length ✅
      - metrics.json: stores sifted length, QBER, reduction, etc.
    """
    # --- metrics
    qber = compute_qber(tx_bits, rx_bits)
    atk_prob = attack_success_probability(compromised_flags)
    reduction = None
    if baseline_attack_prob is not None and baseline_attack_prob > 0:
        reduction = max(0.0, (baseline_attack_prob - atk_prob) / baseline_attack_prob)

    # --- symbolwise CSV (equal-length columns only)
    df = pd.DataFrame(
        {
            "tx": tx_bits.astype(int),                 # length N
            "compromised": compromised_flags.astype(int),  # length N
        }
    )
    df.to_csv(out_csv_path, index=False)

    # --- metrics sidecar JSON (same prefix as CSV)
    metrics_obj = {
        "requested_keys": int(tx_bits.size),
        "sifted_keys": int(rx_bits.size),
        "qber": float(qber),
        "attack_success_prob": float(atk_prob),
        "reduction_vs_baseline": (None if reduction is None else float(reduction)),
        "csv_path": out_csv_path,
    }
    metrics_json_path = out_csv_path.replace(".csv", "_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2)

    return Metrics(
        requested_keys=int(tx_bits.size),
        sifted_keys=int(rx_bits.size),
        qber=float(qber),
        attack_success_prob=float(atk_prob),
        reduction_vs_baseline=(None if reduction is None else float(reduction)),
    )
