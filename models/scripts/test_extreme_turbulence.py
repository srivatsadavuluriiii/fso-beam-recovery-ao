"""Test AO performance under extreme turbulence (Cn² = 10e-12)."""

import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from pipeline import SimulationConfig, run_e2e_simulation

print("="*70)
print("ADAPTIVE OPTICS vs EXTREME TURBULENCE (Cn² = 10e-12)")
print("="*70)
print("\nNote: 10e-12 is EXTREME turbulence (typical: 1e-15 to 1e-14)")
print("This tests AO's ability to handle very strong atmospheric distortion.\n")

# Test WITHOUT AO
print("[1] WITHOUT Adaptive Optics:")
print("-" * 70)
config_no_ao = SimulationConfig()
config_no_ao.ENABLE_AO = False
config_no_ao.CN2 = 10e-12
config_no_ao.N_INFO_BITS = 200 * 8  # More bits for better statistics
config_no_ao.N_GRID = 256
config_no_ao.ADD_NOISE = True
config_no_ao.ENABLE_POWER_PROBE = False

results_no_ao = run_e2e_simulation(config_no_ao, verbose=False)

print(f"\nResults WITHOUT AO:")
print(f"  BER: {results_no_ao['metrics']['ber']:.4e}")
print(f"  Bit Errors: {results_no_ao['metrics']['bit_errors']}/{results_no_ao['metrics']['total_bits']}")
cond_no_ao = results_no_ao['metrics'].get('cond_H', 'N/A')
print(f"  Channel Condition: {cond_no_ao}")

# Test WITH AO
print("\n" + "="*70)
print("[2] WITH Adaptive Optics (Modal Correction, 15 Zernike modes):")
print("-" * 70)
config_ao = SimulationConfig()
config_ao.ENABLE_AO = True
config_ao.AO_METHOD = 'modal'
config_ao.AO_N_ZERNIKE = 15
config_ao.CN2 = 10e-12  # Same extreme turbulence
config_ao.N_INFO_BITS = 200 * 8
config_ao.N_GRID = 256
config_ao.ADD_NOISE = True
config_ao.ENABLE_POWER_PROBE = False

results_ao = run_e2e_simulation(config_ao, verbose=False)

if 'ao_summary' in results_ao:
    ao = results_ao['ao_summary']
    print(f"\nResults WITH AO:")
    print(f"  BER: {results_ao['metrics']['ber']:.4e}")
    print(f"  Bit Errors: {results_ao['metrics']['bit_errors']}/{results_ao['metrics']['total_bits']}")
    print(f"  Mode Purity: {ao['avg_purity_before']:.4f} → {ao['avg_purity_after']:.4f}")
    print(f"  Channel Condition: {ao['avg_cond_before']:.2e} → {ao['avg_cond_after']:.2e}")
    
    # Calculate improvements
    print("\n" + "="*70)
    print("[3] AO IMPROVEMENT SUMMARY:")
    print("-" * 70)
    
    ber_before = results_no_ao['metrics']['ber']
    ber_after = results_ao['metrics']['ber']
    ber_improvement = (ber_before - ber_after) / ber_before * 100 if ber_before > 0 else 0
    
    print(f"  BER: {ber_before:.4e} → {ber_after:.4e}")
    print(f"  BER Improvement: {ber_improvement:+.1f}%")
    
    purity_improvement = (ao['avg_purity_after'] - ao['avg_purity_before']) * 100
    print(f"  Mode Purity Improvement: {purity_improvement:+.2f}%")
    
    if isinstance(cond_no_ao, (int, float)):
        cond_improvement = (1 - ao['avg_cond_after']/cond_no_ao) * 100
        print(f"  Condition Number Improvement: {cond_improvement:+.1f}%")
    
    # Show dominant aberrations
    if 'ao_metrics' in results_ao and results_ao['ao_metrics']:
        print(f"\n  Dominant Aberrations Corrected:")
        first_metrics = results_ao['ao_metrics'][0]
        if 'zernike_coeffs' in first_metrics:
            coeffs = first_metrics['zernike_coeffs']
            sorted_coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
            mode_names = {1: "Piston", 2: "Tip", 3: "Tilt", 4: "Defocus",
                          5: "Astig 45°", 6: "Astig 0°", 7: "Coma y", 8: "Coma x"}
            for j, a_j in sorted_coeffs[:5]:
                name = mode_names.get(j, f"Mode {j}")
                print(f"    {name:12s} (Z{j:2d}): {a_j:8.4f} rad")
    
    print("\n" + "="*70)
    if ber_improvement > 5:
        print("✓ AO provides significant improvement under extreme turbulence!")
    elif ber_improvement > 0:
        print("✓ AO provides modest improvement under extreme turbulence")
    else:
        print("⚠ AO shows minimal improvement - turbulence may be too strong")
    print("="*70)
else:
    print("AO metrics not available")

