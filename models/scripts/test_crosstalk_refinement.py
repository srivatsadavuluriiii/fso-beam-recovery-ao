"""Test refined crosstalk metric to verify it's working correctly."""

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
print("TESTING REFINED CROSSTALK METRIC")
print("="*70)

# Test with AO enabled
print("\n[1] WITH AO: 35 modes, Modal correction")
print("-" * 70)
config_ao = SimulationConfig()
config_ao.ENABLE_AO = True
config_ao.AO_METHOD = 'modal'
config_ao.AO_N_ZERNIKE = 35
config_ao.CN2 = 10e-12
config_ao.N_INFO_BITS = 100 * 8  # Smaller for faster test
config_ao.N_GRID = 256
config_ao.ADD_NOISE = True
config_ao.ENABLE_POWER_PROBE = False

results_ao = run_e2e_simulation(config_ao, verbose=False)

# Extract detailed AO metrics
if 'ao_metrics' in results_ao and len(results_ao['ao_metrics']) > 0:
    # Get first symbol's metrics for detailed analysis
    first_ao = results_ao['ao_metrics'][0]
    
    print("\n  DETAILED CROSSTALK ANALYSIS (first symbol):")
    print(f"    Crosstalk (before): {first_ao.get('crosstalk_dB_before', 'N/A')}")
    print(f"    Crosstalk (after):  {first_ao.get('crosstalk_dB_after', 'N/A')}")
    print(f"    Reduction:          {first_ao.get('crosstalk_reduction_dB', 'N/A')}")
    
    # Check H matrices
    H_before = first_ao.get('H_before', None)
    H_after = first_ao.get('H_after', None)
    
    if H_before is not None and H_after is not None:
        print("\n  CROSSTALK MATRIX ANALYSIS:")
        H_before_abs_sq = np.abs(H_before) ** 2
        H_after_abs_sq = np.abs(H_after) ** 2
        
        diag_before = np.sum(np.diag(H_before_abs_sq))
        diag_after = np.sum(np.diag(H_after_abs_sq))
        off_diag_before = np.sum(H_before_abs_sq) - diag_before
        off_diag_after = np.sum(H_after_abs_sq) - diag_after
        
        print(f"    Diagonal power (before): {diag_before:.6e}")
        print(f"    Diagonal power (after):  {diag_after:.6e}")
        print(f"    Off-diagonal power (before): {off_diag_before:.6e}")
        print(f"    Off-diagonal power (after):  {off_diag_after:.6e}")
        
        if diag_before > 0 and off_diag_before > 0:
            ratio_before = off_diag_before / diag_before
            xt_before_dB = 10 * np.log10(ratio_before)
            print(f"    Crosstalk ratio (before): {ratio_before:.6f} ({xt_before_dB:.2f} dB)")
        
        if diag_after > 0 and off_diag_after > 0:
            ratio_after = off_diag_after / diag_after
            xt_after_dB = 10 * np.log10(ratio_after)
            print(f"    Crosstalk ratio (after):  {ratio_after:.6f} ({xt_after_dB:.2f} dB)")
            
            if diag_before > 0 and off_diag_before > 0:
                reduction = xt_before_dB - xt_after_dB
                print(f"    Crosstalk reduction:     {reduction:+.2f} dB")
        
        # Check matrix structure
        print("\n  MATRIX STRUCTURE:")
        print(f"    H_before shape: {H_before.shape}")
        print(f"    H_after shape:  {H_after.shape}")
        print(f"    Max off-diagonal (before): {np.max(np.abs(H_before - np.diag(np.diag(H_before)))):.6e}")
        print(f"    Max off-diagonal (after):  {np.max(np.abs(H_after - np.diag(np.diag(H_after)))):.6e}")

# Summary from aggregated metrics
if 'ao_summary' in results_ao:
    ao = results_ao['ao_summary']
    print("\n" + "="*70)
    print("AGGREGATED METRICS")
    print("="*70)
    print(f"  Crosstalk (before): {ao.get('avg_crosstalk_dB_before', 'N/A')}")
    print(f"  Crosstalk (after):  {ao.get('avg_crosstalk_dB_after', 'N/A')}")
    print(f"  Reduction:          {ao.get('avg_crosstalk_reduction_dB', 'N/A')} dB")
    print("="*70)

