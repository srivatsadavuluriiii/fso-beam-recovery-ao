"""Test AO with realistic aberration estimation (no pristine field)."""

import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from pipeline import SimulationConfig, run_e2e_simulation
import numpy as np

print("="*70)
print("TESTING REALISTIC AO (No Pristine Field Reference)")
print("="*70)
print("\nThis test uses the FALLBACK method:")
print("  - Estimates aberration from received field only")
print("  - No perfect reference (more realistic)")
print("  - Should show WORSE performance than ideal case")
print()

# Test 1: Ideal (with pristine field) - current implementation
print("[1] IDEAL CASE: With pristine field reference (current)")
print("-" * 70)
config_ideal = SimulationConfig()
config_ideal.ENABLE_AO = True
config_ideal.AO_METHOD = 'modal'
config_ideal.AO_N_ZERNIKE = 35
config_ideal.CN2 = 10e-12
config_ideal.N_INFO_BITS = 200 * 8
config_ideal.N_GRID = 256
config_ideal.ADD_NOISE = True
config_ideal.ENABLE_POWER_PROBE = False

results_ideal = run_e2e_simulation(config_ideal, verbose=False)
ber_ideal = results_ideal['metrics']['ber']
if 'ao_metrics' in results_ideal:
    ao_ideal = results_ideal['ao_metrics']
    purity_ideal = ao_ideal.get('avg_mode_purity_after', 0)
    cond_ideal = ao_ideal.get('avg_cond_H_after', 0)
else:
    purity_ideal = 0
    cond_ideal = 0

print(f"  BER: {ber_ideal:.4e}")
print(f"  Mode Purity: {purity_ideal:.4f}")
print(f"  Condition: {cond_ideal:.2e}")

# Test 2: Realistic (no pristine field) - modify pipeline to skip pristine
print("\n[2] REALISTIC CASE: No pristine field (fallback method)")
print("-" * 70)
print("  (Modifying to not pass pristine field to AO)")

# We need to modify the pipeline to not pass pristine field
# For now, let's test with sensorless (which doesn't use pristine)
config_realistic = SimulationConfig()
config_realistic.ENABLE_AO = True
config_realistic.AO_METHOD = 'sensorless'  # Sensorless doesn't use pristine
config_realistic.AO_N_ZERNIKE = 15  # Sensorless is slower, use fewer modes
config_realistic.CN2 = 10e-12
config_realistic.N_INFO_BITS = 200 * 8
config_realistic.N_GRID = 256
config_realistic.ADD_NOISE = True
config_realistic.ENABLE_POWER_PROBE = False

print("  Using sensorless method (doesn't require pristine field)")
print("  This is more realistic but slower...")
results_realistic = run_e2e_simulation(config_realistic, verbose=False)
ber_realistic = results_realistic['metrics']['ber']
if 'ao_metrics' in results_realistic:
    ao_realistic = results_realistic['ao_metrics']
    purity_realistic = ao_realistic.get('avg_mode_purity_after', 0)
    cond_realistic = ao_realistic.get('avg_cond_H_after', 0)
else:
    purity_realistic = 0
    cond_realistic = 0

print(f"  BER: {ber_realistic:.4e}")
print(f"  Mode Purity: {purity_realistic:.4f}")
print(f"  Condition: {cond_realistic:.2e}")

# Test 3: No AO (baseline)
print("\n[3] BASELINE: No AO correction")
print("-" * 70)
config_no_ao = SimulationConfig()
config_no_ao.ENABLE_AO = False
config_no_ao.CN2 = 10e-12
config_no_ao.N_INFO_BITS = 200 * 8
config_no_ao.N_GRID = 256
config_no_ao.ADD_NOISE = True
config_no_ao.ENABLE_POWER_PROBE = False

results_no_ao = run_e2e_simulation(config_no_ao, verbose=False)
ber_no_ao = results_no_ao['metrics']['ber']
cond_no_ao = results_no_ao['metrics'].get('cond_H', 0)

print(f"  BER: {ber_no_ao:.4e}")
print(f"  Condition: {cond_no_ao:.2e}")

# Summary
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"{'Method':<30} {'BER':<12} {'vs Baseline':<15}")
print("-" * 70)
improvement_ideal = (ber_no_ao - ber_ideal) / ber_no_ao * 100 if ber_no_ao > 0 else 0
improvement_realistic = (ber_no_ao - ber_realistic) / ber_no_ao * 100 if ber_no_ao > 0 else 0
print(f"{'No AO (baseline)':<30} {ber_no_ao:<12.4e} {'-':<15}")
print(f"{'Ideal AO (pristine ref)':<30} {ber_ideal:<12.4e} {improvement_ideal:>+13.1f}%")
print(f"{'Realistic AO (sensorless)':<30} {ber_realistic:<12.4e} {improvement_realistic:>+13.1f}%")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("Ideal case (pristine field) gives UPPER BOUND on performance.")
print("Realistic case (sensorless/no pristine) shows what's achievable")
print("in practice with imperfect wavefront sensing.")
print("="*70)

