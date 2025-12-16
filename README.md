# Tiny World Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17952632.svg)](https://doi.org/10.5281/zenodo.17952632)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Code for **"Inductive Resonance in Gated Recurrent Networks"** (2025).

## Key Findings

| Experiment | World | Model | ΔNLL (struct - null) | p-value |
|------------|-------|-------|---------------------|---------|
| E1 | Drift | GRU | -0.847 ± 0.023 | < 0.001 |
| E1 | Drift | RNN | -0.312 ± 0.019 | < 0.001 |
| E2 | Regime | GRU | -1.234 ± 0.031 | < 0.001 |
| E2 | Regime | RNN | -0.891 ± 0.027 | < 0.001 |
| E3 | Oscillator | GRU | -0.156 ± 0.018 | 0.023 |
| E3 | Oscillator | RNN | -0.089 ± 0.015 | 0.187 |

**Main result:** GRUs develop high-Q resonant manifolds (λ₁ ≈ 0) that amplify temporal structure, while RNNs show broader Lyapunov spectra. This "inductive resonance" explains GRU hallucination on null worlds.

## Installation

```bash
# Clone repository
git clone https://github.com/chimera-sigma/tiny-worlds.git
cd tiny_world_model

# Create environment
conda env create -f environment.yml
conda activate tiny_world

# Or using pip
pip install -r requirements.txt
```

## Quick Start

```bash
# Run single experiment
python -m tiny_world_model.train world=drift model=gru seed=42

# Run full experiment suite (20 seeds x 6 configs)
python scripts/run_experiments.py

# Analyze results
python scripts/analyze_results.py
```

## Project Structure

```
tiny_world_model/
├── configs/                 # Hydra configuration files
│   ├── world/              # World definitions (drift, regime, oscillator)
│   ├── model/              # Model architectures (rnn, gru, lstm)
│   ├── train/              # Training hyperparameters
│   └── time_cond/          # Time conditioning options
├── tiny_world_model/       # Core package
│   ├── models/             # RNN/GRU/LSTM implementations
│   ├── worlds/             # Environment definitions
│   ├── probes.py           # Linear probe analysis
│   └── metrics.py          # ΔNLL, permutation tests
├── experiments/            # Experiment scripts and results
└── paper/                  # LaTeX source for paper
```

## Experiments

### E1: Drift World
Tests whether models distinguish linear temporal drift from IID noise.

```bash
python -m tiny_world_model.train world=drift model=gru
python -m tiny_world_model.train world=drift_null model=gru
```

### E2: Regime World
Tests hidden state inference across regime switches.

```bash
python -m tiny_world_model.train world=regime model=gru
python -m tiny_world_model.train world=regime_null model=gru
```

### E3: Oscillator World
Tests phase tracking in periodic dynamics.

```bash
python -m tiny_world_model.train world=oscillator model=gru
python -m tiny_world_model.train world=oscillator_null model=gru
```

## Statistical Methods

- **20 seeds** per configuration for robust estimation
- **Sign-flip permutation tests** (9999 resamples) for p-values
- **>80% power** at α=0.05 for detecting ΔNLL > 0.1

## Citation

```bibtex
@software{giebelhaus2025inductive,
  title={Inductive Resonance in Gated Recurrent Networks},
  author={Giebelhaus, M. Axel},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17952632},
  url={https://doi.org/10.5281/zenodo.17952632}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Data Availability

Full experiment results available on Zenodo: [10.5281/zenodo.17952632](https://doi.org/10.5281/zenodo.17952632)
