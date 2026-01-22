# DDFS ACC 2026

**Robust Data-Driven Control for Nonlinear Systems Using their Digital Twins and Quadratic Funnels**

Authors: Shiva Shakeri and Mehran Mesbahi
University of Washington

## Overview

This repository implements the data-driven funnel synthesis framework from the ACC 2026 paper. The method:

1. Uses a digital twin to generate a nominal trajectory
2. Collects online data from the physical plant
3. Synthesizes robust quadratic funnels via LMIs
4. Guarantees constraint satisfaction through certified tubes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Project Structure

```
ddfs_acc2026/
├── config/                 # Configuration and parameters
├── dynamics/               # Robot arm dynamics and discretization
├── trajectory/             # Nominal trajectory planning
├── linearization/          # Jacobians and Lipschitz estimation
├── data_driven/            # Data collection and uncertainty sets
├── synthesis/              # LMI matrices and SDP solver
├── control/                # Segmented controller (Algorithm 1)
├── simulation/             # Closed-loop simulation
├── visualization/          # Plotting utilities
├── tests/                  # Unit and integration tests
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Key Parameters

From the paper:
- `dt = 0.01` s (discretization step)
- `N = 600` steps (6 seconds total)
- `T = 100` steps per segment
- `L = 60` steps (data window length)
- `alpha = 0.98` (Lyapunov decay rate)
- `mu = 1.02` (cross-segment growth bound)
- `gamma = 0.034` (mismatch bound)
- `L_J = 16.94` (Jacobian Lipschitz constant)
- `L_r = 0.074` (remainder Lipschitz constant)

## Expected Results

- Terminal deviation: ||η(N)|| ≈ 4.725 × 10⁻²
- 6 segments complete successfully
- All SDPs solve to optimality
- Proposed method outperforms twin-only baseline
