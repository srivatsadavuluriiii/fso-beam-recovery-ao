"""Test improved AO methods: more Zernike modes and sensorless optimization."""

import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from pipeline import SimulationConfig, run_e2e_simulation

print("="*70)
print("TESTING IMPROVED AO METHODS WITH EXTREME TURBULENCE (Cn² = 10e-12)")
print("="*70)

# Baseline: Current setup
print("\n[1] BASELINE: Modal correction, 15 Zernike modes")
print("-" * 70)
config_baseline = SimulationConfig()
config_baseline.ENABLE_AO = True
config_baseline.AO_METHOD = 'modal'
config_baseline.AO_N_ZERNIKE = 15
config_baseline.CN2 = 10e-12
config_baseline.N_INFO_BITS = 200 * 8
config_baseline.N_GRID = 256
config_baseline.ADD_NOISE = True
config_baseline.ENABLE_POWER_PROBE = False

results_baseline = run_e2e_simulation(config_baseline, verbose=False)
ber_baseline = results_baseline['metrics']['ber']
cond_baseline = results_baseline['metrics'].get('cond_H', 0)
if 'ao_summary' in results_baseline:
    ao_baseline = results_baseline['ao_summary']
    purity_baseline = ao_baseline['avg_purity_after']
    cond_ao_baseline = ao_baseline['avg_cond_after']
else:
    purity_baseline = 0
    cond_ao_baseline = 0

print(f"  BER: {ber_baseline:.4e}")
print(f"  Mode Purity: {purity_baseline:.4f}")
print(f"  Condition (after AO): {cond_ao_baseline:.2e}")

# Test 1: More modes (21)
print("\n[2] MORE MODES: Modal correction, 21 Zernike modes")
print("-" * 70)
config_21 = SimulationConfig()
config_21.ENABLE_AO = True
config_21.AO_METHOD = 'modal'
config_21.AO_N_ZERNIKE = 21
config_21.CN2 = 10e-12
config_21.N_INFO_BITS = 200 * 8
config_21.N_GRID = 256
config_21.ADD_NOISE = True
config_21.ENABLE_POWER_PROBE = False

results_21 = run_e2e_simulation(config_21, verbose=False)
ber_21 = results_21['metrics']['ber']
if 'ao_summary' in results_21:
    ao_21 = results_21['ao_summary']
    purity_21 = ao_21['avg_purity_after']
    cond_ao_21 = ao_21['avg_cond_after']
else:
    purity_21 = 0
    cond_ao_21 = 0

print(f"  BER: {ber_21:.4e}")
print(f"  Mode Purity: {purity_21:.4f}")
print(f"  Condition (after AO): {cond_ao_21:.2e}")
improvement_21 = (ber_baseline - ber_21) / ber_baseline * 100 if ber_baseline > 0 else 0
print(f"  BER Improvement: {improvement_21:+.1f}%")

# Test 2: Even more modes (35)
print("\n[3] EVEN MORE MODES: Modal correction, 35 Zernike modes")
print("-" * 70)
config_35 = SimulationConfig()
config_35.ENABLE_AO = True
config_35.AO_METHOD = 'modal'
config_35.AO_N_ZERNIKE = 35
config_35.CN2 = 10e-12
config_35.N_INFO_BITS = 200 * 8
config_35.N_GRID = 256
config_35.ADD_NOISE = True
config_35.ENABLE_POWER_PROBE = False

results_35 = run_e2e_simulation(config_35, verbose=False)
ber_35 = results_35['metrics']['ber']
if 'ao_summary' in results_35:
    ao_35 = results_35['ao_summary']
    purity_35 = ao_35['avg_purity_after']
    cond_ao_35 = ao_35['avg_cond_after']
else:
    purity_35 = 0
    cond_ao_35 = 0

print(f"  BER: {ber_35:.4e}")
print(f"  Mode Purity: {purity_35:.4f}")
print(f"  Condition (after AO): {cond_ao_35:.2e}")
improvement_35 = (ber_baseline - ber_35) / ber_baseline * 100 if ber_baseline > 0 else 0
print(f"  BER Improvement: {improvement_35:+.1f}%")

# Test 3: Sensorless method
print("\n[4] SENSORLESS: Sensorless optimization, 15 modes")
print("-" * 70)
print("  (This may take longer - iteratively optimizes correction)")
config_sensorless = SimulationConfig()
config_sensorless.ENABLE_AO = True
config_sensorless.AO_METHOD = 'sensorless'
config_sensorless.AO_N_ZERNIKE = 15
config_sensorless.CN2 = 10e-12
config_sensorless.N_INFO_BITS = 200 * 8
config_sensorless.N_GRID = 256
config_sensorless.ADD_NOISE = True
config_sensorless.ENABLE_POWER_PROBE = False

results_sensorless = run_e2e_simulation(config_sensorless, verbose=False)
ber_sensorless = results_sensorless['metrics']['ber']
if 'ao_summary' in results_sensorless:
    ao_sensorless = results_sensorless['ao_summary']
    purity_sensorless = ao_sensorless['avg_purity_after']
    cond_ao_sensorless = ao_sensorless['avg_cond_after']
else:
    purity_sensorless = 0
    cond_ao_sensorless = 0

print(f"  BER: {ber_sensorless:.4e}")
print(f"  Mode Purity: {purity_sensorless:.4f}")
print(f"  Condition (after AO): {cond_ao_sensorless:.2e}")
improvement_sensorless = (ber_baseline - ber_sensorless) / ber_baseline * 100 if ber_baseline > 0 else 0
print(f"  BER Improvement: {improvement_sensorless:+.1f}%")

# Summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"{'Method':<30} {'BER':<12} {'Improvement':<12} {'Purity':<10} {'Cond(H)':<10}")
print("-" * 70)
print(f"{'Baseline (15 modes, modal)':<30} {ber_baseline:<12.4e} {'-':<12} {purity_baseline:<10.4f} {cond_ao_baseline:<10.2e}")
print(f"{'21 modes (modal)':<30} {ber_21:<12.4e} {improvement_21:>+10.1f}% {purity_21:<10.4f} {cond_ao_21:<10.2e}")
print(f"{'35 modes (modal)':<30} {ber_35:<12.4e} {improvement_35:>+10.1f}% {purity_35:<10.4f} {cond_ao_35:<10.2e}")
print(f"{'Sensorless (15 modes)':<30} {ber_sensorless:<12.4e} {improvement_sensorless:>+10.1f}% {purity_sensorless:<10.4f} {cond_ao_sensorless:<10.2e}")

# Find best
methods = [
    ('Baseline', ber_baseline, improvement_21),
    ('21 modes', ber_21, improvement_21),
    ('35 modes', ber_35, improvement_35),
    ('Sensorless', ber_sensorless, improvement_sensorless)
]
best = min(methods, key=lambda x: x[1])

print("\n" + "="*70)
print(f"BEST METHOD: {best[0]} (BER = {best[1]:.4e})")
if best[2] > 0:
    print(f"✓ Provides {best[2]:.1f}% improvement over baseline")
else:
    print("⚠ No improvement over baseline")
print("="*70)

