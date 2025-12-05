# FSO Beam Recovery with Adaptive Optics

A comprehensive Python simulation framework for Free-Space Optical (FSO) communication systems using Orbital Angular Momentum (OAM) mode multiplexing with Adaptive Optics (AO) correction for atmospheric turbulence compensation.

## Overview

This project implements an end-to-end simulation of an FSO-OAM communication system that:

- Transmits data using multiple Laguerre-Gaussian (LG) OAM modes simultaneously
- Simulates realistic atmospheric turbulence effects using multi-layer phase screens
- Applies Adaptive Optics correction to mitigate turbulence-induced wavefront distortions
- Demodulates and decodes received signals with channel estimation and equalization
- Provides comprehensive performance metrics including BER, crosstalk, coupling efficiency, and mode purity

The system demonstrates how Adaptive Optics can significantly improve OAM mode separation and communication performance under various turbulence conditions.

## Key Features

### Communication System
- **Multi-mode OAM Transmission**: Supports up to 8 simultaneous OAM modes (l = ±1, ±3, ±4, p = 0,1)
- **QPSK Modulation**: Quadrature Phase Shift Keying for data encoding
- **LDPC Forward Error Correction**: Rate-adaptive LDPC coding for error recovery
- **Pilot-Based Channel Estimation**: Least-squares channel matrix estimation
- **MMSE Equalization**: Minimum Mean Square Error equalization for interference mitigation

### Atmospheric Turbulence Modeling
- **Multi-Layer Phase Screens**: Realistic turbulence simulation using split-step propagation
- **Configurable Turbulence Strength**: Supports Cn² values from 10⁻¹⁸ to 10⁻¹² m⁻²/³
- **Fried Parameter Calculation**: Coherence diameter (r₀) computation for turbulence characterization
- **Rytov Variance**: Weak-to-strong turbulence regime modeling

### Adaptive Optics Correction
- **Zernike Polynomial Decomposition**: Wavefront aberration representation using orthogonal basis functions
- **Modal Control**: Fast correction using Zernike mode coefficients
- **Sensorless Optimization**: Iterative optimization without explicit wavefront sensing
- **Mode Purity Sensing**: OAM-specific wavefront sensing using mode projection
- **Phase Retrieval**: Alternative sensing method using intensity measurements

### Performance Analysis
- **Comprehensive Metrics**: BER, crosstalk, coupling efficiency, mode purity, condition number
- **Turbulence Sweep Analysis**: Performance evaluation across turbulence levels
- **Visualization Tools**: Comparative plots for before/after AO correction
- **Channel Matrix Analysis**: H_est matrix visualization and condition number tracking

## System Architecture

```
┌─────────────────┐
│   Transmitter   │
│  - OAM Modes    │
│  - QPSK Mod     │   
│  - LDPC Encoder │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Atmospheric    │
│  Channel        │
│  - Turbulence   │
│  - Attenuation  │
│  - Noise        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  Adaptive       │      │   Receiver      │
│  Optics         │─────▶│  - Demux        │
│  - Sensing      │      │  - Channel Est  │
│  - Correction   │      │  - Equalization │
└─────────────────┘      │  - LDPC Decoder │
                         └─────────────────┘
```

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0
- pyldpc >= 0.2.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fso-beam-recovery-ao.git
cd fso-beam-recovery-ao
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to the models directory:
```bash
cd models
```

## Usage

### Basic Simulation

Run a single simulation with default parameters:

```python
from pipeline import SimulationConfig, run_e2e_simulation

config = SimulationConfig()
results = run_e2e_simulation(config, verbose=True)
```

### Enable Adaptive Optics

To enable AO correction:

```python
config = SimulationConfig()
config.ENABLE_AO = True
config.AO_METHOD = 'modal'  # or 'sensorless'
config.AO_N_ZERNIKE = 21    # Number of Zernike modes
results = run_e2e_simulation(config)
```

### Turbulence Sweep Analysis

Compare performance with and without AO across turbulence levels:

```python
from sweep_cn2_ao_comparison import sweep_cn2_with_ao_comparison

cn2_values = [1e-18, 1e-17, 1e-15, 1e-13, 1e-12]
results = sweep_cn2_with_ao_comparison(
    cn2_values=cn2_values,
    n_iterations=5,
    save_plots=True
)
```

### Generate Visualizations

Create comparison plots for AO performance:

```python
from generate_ao_visualizations import generate_all_ao_plots

generate_all_ao_plots(
    cn2=1e-15,
    n_iterations=5,
    save_dir='plots/ao_comparison'
)
```

## Configuration Parameters

### Optical Parameters
- `WAVELENGTH`: Operating wavelength (default: 1550 nm)
- `W0`: Beam waist radius at transmitter (default: 25 mm)
- `DISTANCE`: Link distance (default: 1000 m)
- `RECEIVER_DIAMETER`: Receiver aperture diameter (default: 0.5 m)

### Turbulence Parameters
- `CN2`: Refractive index structure constant (default: 1e-15 m⁻²/³)
- `L0`: Outer scale of turbulence (default: 10.0 m)
- `L0_INNER`: Inner scale of turbulence (default: 0.005 m)
- `NUM_SCREENS`: Number of phase screens for multi-layer propagation (default: 25)

### Communication Parameters
- `SPATIAL_MODES`: List of (p, l) tuples for LG modes
- `FEC_RATE`: LDPC code rate (default: 0.8)
- `PILOT_RATIO`: Fraction of symbols used for pilots (default: 0.1)
- `SNR_DB`: Signal-to-noise ratio (default: 35 dB)

### Adaptive Optics Parameters
- `ENABLE_AO`: Enable/disable AO correction (default: False)
- `AO_METHOD`: Correction method - 'modal' or 'sensorless' (default: 'modal')
- `AO_N_ZERNIKE`: Number of Zernike modes for modal correction (default: 15)

## Results and Performance

### Typical Performance Improvements with AO

Under moderate turbulence (Cn² = 10⁻¹⁵ m⁻²/³):

- **Coupling Efficiency**: 48% → 81% (65% improvement)
- **Crosstalk Reduction**: 2-5 dB improvement
- **Mode Purity**: Significant recovery of mode orthogonality
- **BER Improvement**: 10-20% reduction in bit error rate

### Performance Across Turbulence Levels

![Cn² Sweep Comparison](models/plots/cn2_sweep/cn2_sweep_comparison.png)

The system demonstrates that Adaptive Optics is most effective under moderate-to-strong turbulence conditions, where it provides significant improvements in mode coupling and power recovery.

### AO Improvement Summary

![AO Improvements](models/plots/cn2_sweep/cn2_sweep_improvements.png)

Key findings:
- AO effectiveness peaks at moderate turbulence (Cn² ≈ 10⁻¹⁵)
- Consistent crosstalk reduction across all turbulence levels
- Significant coupling efficiency recovery (48% → 81%)
- Modest but realistic BER improvements (5-15%)

## Project Structure

```
.
├── models/
│   ├── adaptiveOptics.py      # AO correction algorithms
│   ├── encoding.py             # Transmitter and modulation
│   ├── receiver.py             # Receiver and demodulation
│   ├── lgBeam.py               # Laguerre-Gaussian beam generation
│   ├── turbulence.py           # Atmospheric turbulence modeling
│   ├── fsplAtmAttenuation.py  # Path loss and attenuation
│   ├── pipeline.py             # Main simulation pipeline
│   ├── runner.py               # Command-line interface
│   ├── sweep_cn2_ao_comparison.py  # Turbulence sweep analysis
│   ├── visualize_ao_comparison.py  # Visualization utilities
│   └── plots/                  # Generated plots and figures
│       ├── ao_comparison/      # AO performance comparisons
│       └── cn2_sweep/          # Turbulence sweep results
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Algorithms

### Zernike Polynomial Decomposition

Wavefront aberrations are decomposed into Zernike polynomials using least-squares fitting:

```
φ(x,y) = Σᵢ aᵢ Zᵢ(x,y)
```

where Zᵢ are normalized Zernike polynomials and aᵢ are coefficients determined via least-squares minimization.

### Modal Control

The modal controller estimates aberration phase by comparing received field to pristine reference:

```
φ_aberration = arg(received_field) - arg(pristine_field)
```

Zernike coefficients are then computed and applied as correction.

### Channel Estimation

Least-squares channel matrix estimation using pilot symbols:

```
H_est = Y_pilot X_pilot^H (X_pilot X_pilot^H)^(-1)
```

where Y_pilot and X_pilot are received and transmitted pilot symbols.

## Literature References

- **Noll (1976)**: Zernike polynomials for atmospheric turbulence representation
- **Hardy (1998)**: Adaptive Optics for Astronomical Telescopes
- **Booth (2014)**: Wavefront sensorless adaptive optics
- **Andrews & Phillips (2005)**: Laser Beam Propagation Through Random Media
- **Wang et al. (2012)**: OAM mode sensitivity to turbulence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fso_beam_recovery_ao,
  title = {FSO Beam Recovery with Adaptive Optics},
  author = {Srivatsa Davuluri},
  year = {2024},
  url = {https://github.com/srivatsadavuluriiii/fso-beam-recovery-ao}
}
```

## Contact

For questions or inquiries, please open an issue on GitHub.

## Acknowledgments

This work builds upon established methods in adaptive optics, free-space optical communication, and orbital angular momentum multiplexing.

