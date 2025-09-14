"""
visualization.py
Plotting utilities for results.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(values, title, xlabel, outfile):
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def plot_compromise_rates(flags_before, flags_after, outfile):
    def rate(flags):
        return np.mean(flags.astype(np.float32)) if len(flags) else 0.0

    rates = [rate(flags_before), rate(flags_after)]
    labels = ["Baseline", "With COW/Decoys"]
    plt.figure()
    plt.bar(labels, rates)
    plt.title("PNS Compromise Rate")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
