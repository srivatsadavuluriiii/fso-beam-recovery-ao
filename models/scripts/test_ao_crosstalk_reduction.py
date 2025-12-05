"""Compare H_est with/without AO to measure actual crosstalk reduction."""

import sys
import os
import numpy as np

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from pipeline import SimulationConfig, run_e2e_simulation
from adaptiveOptics import ModePuritySensor

print("="*70)
print("COMPARING H_est WITH/WITHOUT AO - CROSSTALK REDUCTION")
print("="*70)
print("\nThis test measures actual crosstalk reduction by comparing")
print("the receiver's H_est matrix with and without AO correction.")
print()

# Test configuration
config_base = SimulationConfig()
config_base.CN2 = 10e-12  # Extreme turbulence
config_base.N_INFO_BITS = 200 * 8
config_base.N_GRID = 256
config_base.ADD_NOISE = True
config_base.ENABLE_POWER_PROBE = False

# Test 1: WITHOUT AO
print("[1] WITHOUT AO (Baseline)")
print("-" * 70)
config_no_ao = config_base
config_no_ao.ENABLE_AO = False

results_no_ao = run_e2e_simulation(config_no_ao, verbose=False)
H_est_no_ao = results_no_ao['metrics'].get('H_est', None)

if H_est_no_ao is None:
    print("ERROR: H_est not found in results!")
    sys.exit(1)

# Compute metrics from H_est
sensor = ModePuritySensor(config_base.SPATIAL_MODES, config_base.WAVELENGTH, 
                         config_base.W0, config_base.DISTANCE)

crosstalk_no_ao = sensor.compute_crosstalk_dB(H_est_no_ao)
coupling_no_ao = sensor.compute_coupling_efficiency(H_est_no_ao)
cond_no_ao = sensor.compute_condition_metric(H_est_no_ao)
purity_no_ao = sensor.compute_purity_metric(H_est_no_ao)
ber_no_ao = results_no_ao['metrics']['ber']

print(f"  BER: {ber_no_ao:.4e}")
print(f"  Crosstalk (H_est): {crosstalk_no_ao:.2f} dB")
print(f"  Coupling Efficiency: {coupling_no_ao:.4f}")
print(f"  Condition Number: {cond_no_ao:.2e}")
print(f"  Mode Purity: {purity_no_ao:.4f}")

# Test 2: WITH AO (35 modes)
print("\n[2] WITH AO: 35 Zernike modes, Modal correction")
print("-" * 70)
config_ao = config_base
config_ao.ENABLE_AO = True
config_ao.AO_METHOD = 'modal'
config_ao.AO_N_ZERNIKE = 35

results_ao = run_e2e_simulation(config_ao, verbose=False)
H_est_ao = results_ao['metrics'].get('H_est', None)

if H_est_ao is None:
    print("ERROR: H_est not found in results!")
    sys.exit(1)

# Compute metrics from H_est
crosstalk_ao = sensor.compute_crosstalk_dB(H_est_ao)
coupling_ao = sensor.compute_coupling_efficiency(H_est_ao)
cond_ao = sensor.compute_condition_metric(H_est_ao)
purity_ao = sensor.compute_purity_metric(H_est_ao)
ber_ao = results_ao['metrics']['ber']

print(f"  BER: {ber_ao:.4e}")
print(f"  Crosstalk (H_est): {crosstalk_ao:.2f} dB")
print(f"  Coupling Efficiency: {coupling_ao:.4f}")
print(f"  Condition Number: {cond_ao:.2e}")
print(f"  Mode Purity: {purity_ao:.4f}")

# Calculate improvements
crosstalk_reduction = crosstalk_no_ao - crosstalk_ao  # Positive = reduction
coupling_improvement = coupling_ao - coupling_no_ao
cond_improvement = (cond_no_ao - cond_ao) / cond_no_ao * 100  # % reduction
purity_improvement = purity_ao - purity_no_ao
ber_improvement = (ber_no_ao - ber_ao) / ber_no_ao * 100  # % reduction

# Summary
print("\n" + "="*70)
print("CROSSTALK REDUCTION ANALYSIS")
print("="*70)
print(f"{'Metric':<25} {'No AO':<15} {'With AO':<15} {'Improvement':<15}")
print("-" * 70)
print(f"{'Crosstalk (dB)':<25} {crosstalk_no_ao:<15.2f} {crosstalk_ao:<15.2f} {crosstalk_reduction:>+13.2f} dB")
print(f"{'Coupling Efficiency':<25} {coupling_no_ao:<15.4f} {coupling_ao:<15.4f} {coupling_improvement:>+13.4f}")
print(f"{'Condition Number':<25} {cond_no_ao:<15.2e} {cond_ao:<15.2e} {cond_improvement:>+13.1f}%")
print(f"{'Mode Purity':<25} {purity_no_ao:<15.4f} {purity_ao:<15.4f} {purity_improvement:>+13.4f}")
print(f"{'BER':<25} {ber_no_ao:<15.4e} {ber_ao:<15.4e} {ber_improvement:>+13.1f}%")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
if crosstalk_reduction > 0:
    print(f"✅ Crosstalk REDUCED by {crosstalk_reduction:.2f} dB")
    print(f"   (Literature reports: 10-18 dB reduction under strong turbulence)")
else:
    print(f"⚠️  Crosstalk INCREASED by {abs(crosstalk_reduction):.2f} dB")
    print(f"   (This may indicate AO is not effectively reducing mode coupling)")

if cond_improvement > 0:
    print(f"✅ Condition Number IMPROVED by {cond_improvement:.1f}%")
else:
    print(f"⚠️  Condition Number WORSENED by {abs(cond_improvement):.1f}%")

if ber_improvement > 0:
    print(f"✅ BER IMPROVED by {ber_improvement:.1f}%")
else:
    print(f"⚠️  BER WORSENED by {abs(ber_improvement):.1f}%")

# Matrix analysis
print("\n" + "="*70)
print("H_est MATRIX ANALYSIS")
print("="*70)
H_no_ao_abs_sq = np.abs(H_est_no_ao) ** 2
H_ao_abs_sq = np.abs(H_est_ao) ** 2

diag_no_ao = np.sum(np.diag(H_no_ao_abs_sq))
diag_ao = np.sum(np.diag(H_ao_abs_sq))
off_diag_no_ao = np.sum(H_no_ao_abs_sq) - diag_no_ao
off_diag_ao = np.sum(H_ao_abs_sq) - diag_ao

print(f"Diagonal power (No AO):    {diag_no_ao:.6e}")
print(f"Diagonal power (With AO):  {diag_ao:.6e}")
print(f"Off-diagonal power (No AO):  {off_diag_no_ao:.6e}")
print(f"Off-diagonal power (With AO): {off_diag_ao:.6e}")

diag_improvement = (diag_ao - diag_no_ao) / diag_no_ao * 100 if diag_no_ao > 0 else 0
off_diag_reduction = (off_diag_no_ao - off_diag_ao) / off_diag_no_ao * 100 if off_diag_no_ao > 0 else 0

print(f"\nDiagonal power change:     {diag_improvement:+.1f}%")
print(f"Off-diagonal power reduction: {off_diag_reduction:+.1f}%")

# Check if off-diagonal elements are actually reduced
print("\n" + "="*70)
print("VALIDATION AGAINST LITERATURE")
print("="*70)
print("Literature benchmarks for strong turbulence (D/r₀ ≈ 15-17):")
print("  - Crosstalk reduction: 10-18 dB")
print("  - Coupling efficiency: 50-68%")
print("  - BER improvements: Modest under extreme turbulence")
print()
if crosstalk_reduction > 0:
    if crosstalk_reduction >= 10:
        print(f"✅ EXCELLENT: {crosstalk_reduction:.2f} dB reduction matches literature!")
    elif crosstalk_reduction >= 5:
        print(f"✅ GOOD: {crosstalk_reduction:.2f} dB reduction (moderate improvement)")
    else:
        print(f"⚠️  MODEST: {crosstalk_reduction:.2f} dB reduction (less than literature)")
else:
    print(f"❌ NO IMPROVEMENT: Crosstalk increased by {abs(crosstalk_reduction):.2f} dB")
print("="*70)

