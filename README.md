# Coherent One-Way QKD with Quantum Random Number Generation

This repository contains the implementation of a **Coherent One-Way (COW) Quantum Key Distribution (QKD) protocol** integrated with a **Quantum Random Number Generator (QRNG)**.
The simulation evaluates the protocol’s resilience against **Photon-Number Splitting (PNS) attacks** and demonstrates a **70% reduction in attack success probability** compared to a naive baseline (see `notebooks/Attack_Resilience.ipynb`).

> **Claim context**: This repo includes a reproducible pipeline and scripts to regenerate the numbers and plots that backed the summary:
> *10,000+ secure keys generated* and *~70% reduction in PNS attack success probability* under the simulation assumptions herein.

---

## 🚀 Features
- Simulation of the **COW QKD protocol** with weak coherent pulses.
- **QRNG** module for unbiased bit/basis generation (with simple statistical sanity checks).
- **PNS attack** modeling with configurable Eve capabilities.
- End-to-end pipeline: generate pulses → channel + detectors → sifting → analysis → plots.
- Reproducible **notebooks** and **unit tests**.

## 🛠️ Installation
```bash
git clone https://github.com/Sharika224/COW-QKD-QRNG-Simulation.git
cd COW-QKD-QRNG-Simulation
pip install -r requirements.txt
