# Decoupled Predictive Coding Networks for Backpropagation Benchmarking  
### PhD Research Proposal Repository (2025)
#### Author: Mark A. Griffiths (markalfredgriffiths@hotmail.com, mark.alfred.griffiths@kcl.ac.uk)
---

## Overview

This repository contains research code exploring **decoupled predictive coding (PC)** approaches as alternatives and complements to conventional backpropagation-based learning.

The project benchmarks predictive-coding-inspired learning dynamics against standard gradient-based optimization under:

- clean training conditions,
- structured perturbation regimes,
- and multiple forms of input corruption and noise.

The broader aim is to investigate whether biologically inspired inference and credit-assignment mechanisms can achieve competitive optimization behavior while offering advantages in robustness, locality, parallelism, or scalability.

The repository is designed as a **research benchmarking framework** rather than a production ML library.

---

## Research Motivation

Backpropagation remains the dominant paradigm for training deep neural networks, yet it possesses several characteristics that are often viewed as biologically implausible, including:

- strict weight symmetry,
- tightly coupled forward/backward passes,
- global gradient propagation,
- and synchronization constraints across layers.

Predictive coding frameworks offer an alternative perspective in which learning emerges through iterative inference dynamics and local error minimization.

This project investigates whether **decoupled predictive coding architectures** can:

- approximate useful learning signals,
- support scalable optimization,
- improve robustness under perturbation,
- and potentially enable more parallelized or biologically plausible learning mechanisms.

The work additionally explores the use of:

- synthetic-gradient-style decoupling,
- amortized inference,
- predictive coding energy minimization,
- and path-integral-based update estimation.

---

## Repository Goals

This codebase exists to support:

- empirical benchmarking,
- controlled ablation studies,
- reproducible experimentation,
- and comparative evaluation across learning paradigms.

The repository is intended for:

- researchers working on biologically inspired learning,
- machine learning practitioners exploring alternative optimization methods,
- and collaborators reviewing or extending the framework.

---

# Core Concepts

The repository centers around comparisons between:

| Method | Description |
|---|---|
| Backpropagation | Conventional gradient-based optimization |
| Predictive Coding (PC) | Iterative inference-based learning dynamics |
| Decoupled PC | Predictive coding with decoupled or amortized synthetic gradient estimation |

The decoupled framework explores mechanisms inspired by:

- synthetic gradients,
- amortized inference,
- predictive coding energy minimization,
- and path-integral-based update estimation.

---

## External Dependency: JPC

This repository intentionally does **not** vendor the upstream `jpc` source code.

The predictive coding implementation is treated as an external dependency.

Install directly from upstream:

```bash
pip install git+https://github.com/thebuckleylab/jpc.git
```

For strict reproducibility, pin to a specific commit or release tag.

---

## JPC Compatibility Notes

Recent versions of `jpc` introduced API renaming changes.

If you encounter errors such as:

```python
ImportError: cannot import name 'neg_activity_grad' from 'jpc'
```

this likely reflects upstream API changes.

### Example Migration

Old API:

```python
from jpc import neg_activity_grad
```

New API:

```python
from jpc import neg_pc_activity_grad
```

Similarly:

| Legacy Name | Newer Name |
|---|---|
| `neg_activity_grad` | `neg_pc_activity_grad` |
| `update_activities` | `update_pc_activities` |

Always verify compatibility against the installed upstream JPC version.

---

# Repository Structure

## Core Training and Benchmarking

```text
core/
├── main_adapt.py
├── backprop_implementation.py
├── pc_mnist_baseline.py
└── pc_cifar10_baseline.py
```

### Key Entry Points

| File | Purpose |
|---|---|
| `core/main_adapt.py` | Primary orchestration and experiment entry point |
| `core/backprop_implementation.py` | Baseline backpropagation implementation |
| `core/pc_mnist_baseline.py` | Predictive coding baseline on MNIST |
| `core/pc_cifar10_baseline.py` | Predictive coding baseline on CIFAR-10 |

---

## Benchmarking and Analysis

```text
scripts/analysis/benchmarking/
└── backprop_vs_pc_vs_ilasgn.py
```

Additional utilities:

```text
scripts/analysis/
├── pc_testing/
│   └── compare_lambda_caps.py
├── sweep_path_integral_grid.py
```

---

## Plotting Utilities

```text
scripts/plotting/
├── benchmarking/
│   └── plot_noise_robustness.py
├── plot_seed_robustness.py
├── pc_testing/
│   ├── plot_pc_equilibrium.py
│   └── plot_lambda_gate_vs_pi.py
```

---

## Create Environment

```bash
conda create -n decoupled-pc python=3.11
conda activate decoupled-pc
```

## Install Dependencies

```bash
pip install -r constraints.txt
```

This installs:

- the pinned JAX stack,
- CUDA-compatible JAX plugins,
- and the pinned upstream JPC implementation.

---

## Environment Notes

The provided dependency constraints target CUDA 12 compatible systems.

Depending on your platform and accelerator configuration, you may need to adjust:

- CUDA toolkit version,
- NVIDIA driver version,
- or JAX CUDA compatibility.

For CPU-only environments, alternative JAX installation variants may be preferable.

---

## Dependency Source

Primary dependencies are defined in:

```text
constraints.txt
```

Example contents:

```text
jax==0.5.2
jax-cuda12-pjrt==0.5.1
jax-cuda12-plugin==0.5.1
jaxlib==0.5.1
jaxtyping==0.3.3
jpc @ git+https://github.com/thebuckleylab/jpc.git@64eeab0334fe7e1a4365da5b7f02afaefecd0261
```

The JPC dependency is pinned to a specific upstream commit to improve API stability and experimental reproducibility.
---

# Quickstart

Run the primary experimental pipeline:

```bash
python core/main_adapt.py
```

This serves as the canonical starting point for reproducing baseline comparisons and running benchmark experiments.

---

# Reproducing Benchmarks

For reproducible experimental evaluation, record:

- Python version,
- operating system,
- CUDA/JAX versions,
- accelerator hardware,
- random seeds,
- and exact execution commands.

Recommended practices:

- version-control experiment configurations,
- separate exploratory and publication-grade runs,
- preserve run metadata,
- and archive generated figures alongside experiment manifests.

---

# Noise Robustness Evaluation

The repository includes benchmarking utilities for evaluating performance under perturbation regimes including:

- Gaussian noise,
- salt-and-pepper corruption,
- and structured occlusion.

These analyses are intended to evaluate comparative robustness between:

- backpropagation,
- standard predictive coding,
- and decoupled predictive coding variants.

---

## Run configuration scripts

Use these shell scripts to run the two benchmark suites:

- Number of PC steps to equilibrium: `run_configs/pc_testing/run_pc_testing.sh` 
 
- Robustness to noise testing: `run_configs/main/run_robustness_to_noise_testing.sh`

Examples:

```bash
bash run_configs/pc_testing/run_pc_testing.sh
bash run_configs/main/run_robustness_to_noise_testing.sh
```

# Data and Artifact Management

To keep repository size manageable, large outputs should remain outside default version control.

## Recommended to Keep in Repository

- source code,
- plotting scripts,
- analysis utilities,
- configuration files,
- documentation.

## Recommended to Exclude

```text
data/
pi_plots_out/
backprop_runs/
pc_runs/
ilasgn_runs/
logs/
scratch/
```

Large artifacts are better managed through:

- release attachments,
- cloud artifact storage,
- or external archival services.

---

# Intended Use

This repository is an evolving research framework intended for:

- benchmarking,
- experimentation,
- and methodological exploration.

Interfaces and implementations may change as experiments mature into publication-quality studies.

Users are encouraged to:

- pin dependencies,
- validate upstream compatibility,
- and preserve experiment manifests for longitudinal comparison.

---

# Future Directions

Planned and exploratory extensions include:

- uncertainty-aware predictive coding,
- path-integral-informed synthetic gradients,
- structure learning,
- stochastic latent inference,
- and continuous-time control formulations inspired by Kalman-Bucy filtering and active inference frameworks.

These components are currently exploratory and may not yet be fully represented in the public codebase.

---

# Suggested Repository Name

```text
decoupled-predictive-coding-benchmarks
```

---

# Citation

Citation metadata and BibTeX entries will be added once accompanying thesis or manuscript outputs are finalized.

---

# Contact

Add contact details, institutional affiliation, website, or LinkedIn information here when ready.

---

# License

This repository is provided for research, review, and educational purposes only.

Commercial use is prohibited without explicit written permission from the author.

No warranty is provided.

Copyright (c) 2025 Mark Griffiths.
All rights reserved.
