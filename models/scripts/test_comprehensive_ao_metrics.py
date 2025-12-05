"""Test AO with comprehensive metrics (coupling efficiency, crosstalk, phase error)."""

import sys
import os
import numpy as np

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from pipeline import SimulationConfig, run_e2e_simulation

print("="*70)
print("TESTING AO WITH COMPREHENSIVE METRICS")
print("="*70)
print("\nMetrics being tested:")
print("  - Coupling Efficiency (CE): Power in correct mode")
print("  - Crosstalk (dB): Off-diagonal power / diagonal power")
print("  - Power Loss (dB): Total power before/after")
print("  - Phase Error RMS: Residual phase error")
print()

# Test with 35 modes (best performance)
print("[1] AO ENABLED: 35 Zernike modes, Modal correction")
print("-" * 70)
config_ao = SimulationConfig()
config_ao.ENABLE_AO = True
config_ao.AO_METHOD = 'modal'
config_ao.AO_N_ZERNIKE = 35
config_ao.CN2 = 10e-12
config_ao.N_INFO_BITS = 200 * 8
config_ao.N_GRID = 256
config_ao.ADD_NOISE = True
config_ao.ENABLE_POWER_PROBE = False

results_ao = run_e2e_simulation(config_ao, verbose=False)
ber_ao = results_ao['metrics']['ber']

# Extract comprehensive AO metrics
if 'ao_summary' in results_ao:
    ao = results_ao['ao_summary']
    print(f"\n  BER: {ber_ao:.4e}")
    print(f"\n  COMPREHENSIVE METRICS:")
    print(f"    Mode Purity: {ao.get('avg_purity_before', 0):.4f} → {ao.get('avg_purity_after', 0):.4f}")
    print(f"    Condition Number: {ao.get('avg_cond_before', 0):.2e} → {ao.get('avg_cond_after', 0):.2e}")
    
    ce_before = ao.get('avg_coupling_efficiency_before', 0)
    ce_after = ao.get('avg_coupling_efficiency_after', 0)
    if ce_before > 0:
        print(f"    Coupling Efficiency: {ce_before:.4f} → {ce_after:.4f} "
              f"(+{(ce_after-ce_before)*100:.2f}%)")
    
    xt_before = ao.get('avg_crosstalk_dB_before', np.nan)
    xt_after = ao.get('avg_crosstalk_dB_after', np.nan)
    xt_reduction = ao.get('avg_crosstalk_reduction_dB', 0)
    if np.isfinite(xt_before) and np.isfinite(xt_after):
        print(f"    Crosstalk: {xt_before:.2f} dB → {xt_after:.2f} dB "
              f"(reduction: {xt_reduction:+.2f} dB)")
    
    power_loss = ao.get('avg_power_loss_dB', 0)
    print(f"    Power Loss: {power_loss:.2f} dB")
    
    phase_before = ao.get('avg_phase_error_rms_before', None)
    phase_after = ao.get('avg_phase_error_rms_after', None)
    phase_reduction = ao.get('avg_phase_error_reduction_rad', None)
    if phase_before is not None and phase_after is not None:
        print(f"    Phase Error RMS: {phase_before:.4f} rad → {phase_after:.4f} rad")
        if phase_reduction is not None:
            print(f"      Reduction: {phase_reduction:.4f} rad ({phase_reduction/phase_before*100:.1f}%)")
else:
    print("  No AO metrics available")

# Test without AO (baseline)
print("\n[2] BASELINE: No AO correction")
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
print(f"  Condition Number: {cond_no_ao:.2e}")

# Summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
improvement = (ber_no_ao - ber_ao) / ber_no_ao * 100 if ber_no_ao > 0 else 0
print(f"{'Metric':<30} {'No AO':<15} {'With AO':<15} {'Improvement':<15}")
print("-" * 70)
print(f"{'BER':<30} {ber_no_ao:<15.4e} {ber_ao:<15.4e} {improvement:>+13.1f}%")
print(f"{'Condition Number':<30} {cond_no_ao:<15.2e} {ao.get('avg_cond_after', 0):<15.2e} "
      f"{(cond_no_ao - ao.get('avg_cond_after', cond_no_ao))/cond_no_ao*100:>+13.1f}%")

if 'avg_crosstalk_reduction_dB' in ao and np.isfinite(ao['avg_crosstalk_reduction_dB']):
    print(f"\n  Crosstalk Reduction: {ao['avg_crosstalk_reduction_dB']:+.2f} dB")
    print(f"  (Literature reports: 10-18 dB reduction)")

if 'avg_phase_error_reduction_rad' in ao and ao['avg_phase_error_reduction_rad'] is not None:
    print(f"  Phase Error Reduction: {ao['avg_phase_error_reduction_rad']:.4f} rad RMS")

print("\n" + "="*70)
print("VALIDATION AGAINST LITERATURE")
print("="*70)
print("Literature benchmarks for strong turbulence (D/r₀ ≈ 15-17):")
print("  - Crosstalk reduction: 10-18 dB")
print("  - Coupling efficiency: 50-68%")
print("  - BER improvements: Modest under extreme turbulence")
print("="*70)

