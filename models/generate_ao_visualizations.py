"""
Generate comprehensive AO comparison visualizations.

Runs simulations with and without AO, then creates all comparison plots.
"""

import os
import sys
import numpy as np

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

from pipeline import SimulationConfig, run_e2e_simulation
from visualize_ao_comparison import create_comprehensive_ao_report

def generate_ao_comparison_visualizations(cn2=10e-12, n_zernike=35, output_dir='plots/ao_comparison'):
    """
    Generate comprehensive AO comparison visualizations.
    
    Args:
        cn2: Turbulence strength (m^-2/3)
        n_zernike: Number of Zernike modes for AO
        output_dir: Output directory for plots
    """
    print("="*70)
    print("GENERATING ADAPTIVE OPTICS COMPARISON VISUALIZATIONS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Turbulence: Cn² = {cn2:.1e} m⁻²/³")
    print(f"  Zernike modes: {n_zernike}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Configuration
    config_base = SimulationConfig()
    config_base.CN2 = cn2
    config_base.N_INFO_BITS = 200 * 8  # Smaller for faster runs
    config_base.N_GRID = 256  # Smaller grid for faster runs
    config_base.ADD_NOISE = True
    config_base.ENABLE_POWER_PROBE = False  # Disable for speed
    
    # Run 1: WITHOUT AO
    print("[1/2] Running simulation WITHOUT AO...")
    print("-" * 70)
    config_no_ao = config_base
    config_no_ao.ENABLE_AO = False
    
    results_no_ao = run_e2e_simulation(config_no_ao, verbose=True)
    
    if results_no_ao is None:
        print("✗ Simulation without AO failed!")
        return None
    
    print("\n✓ Simulation without AO completed")
    print(f"  BER: {results_no_ao['metrics']['ber']:.4e}")
    
    # Run 2: WITH AO
    print("\n[2/2] Running simulation WITH AO...")
    print("-" * 70)
    config_ao = config_base
    config_ao.ENABLE_AO = True
    config_ao.AO_METHOD = 'modal'
    config_ao.AO_N_ZERNIKE = n_zernike
    
    results_ao = run_e2e_simulation(config_ao, verbose=True)
    
    if results_ao is None:
        print("✗ Simulation with AO failed!")
        return None
    
    print("\n✓ Simulation with AO completed")
    print(f"  BER: {results_ao['metrics']['ber']:.4e}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_path = create_comprehensive_ao_report(results_no_ao, results_ao, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    from adaptiveOptics import ModePuritySensor
    sensor = ModePuritySensor(config_base.SPATIAL_MODES, config_base.WAVELENGTH, 
                             config_base.W0, config_base.DISTANCE)
    
    H_no_ao = results_no_ao['metrics']['H_est']
    H_ao = results_ao['metrics']['H_est']
    
    crosstalk_no_ao = sensor.compute_crosstalk_dB(H_no_ao)
    crosstalk_ao = sensor.compute_crosstalk_dB(H_ao)
    coupling_no_ao = sensor.compute_coupling_efficiency(H_no_ao)
    coupling_ao = sensor.compute_coupling_efficiency(H_ao)
    ber_no_ao = results_no_ao['metrics']['ber']
    ber_ao = results_ao['metrics']['ber']
    
    print(f"\nCrosstalk Reduction: {crosstalk_no_ao:.2f} dB → {crosstalk_ao:.2f} dB "
          f"({crosstalk_no_ao - crosstalk_ao:+.2f} dB improvement)")
    print(f"Coupling Efficiency: {coupling_no_ao:.4f} → {coupling_ao:.4f} "
          f"({(coupling_ao - coupling_no_ao) / coupling_no_ao * 100:+.1f}% improvement)")
    print(f"BER: {ber_no_ao:.4e} → {ber_ao:.4e} "
          f"({(ber_no_ao - ber_ao) / ber_no_ao * 100:+.1f}% improvement)")
    
    print(f"\n✓ All visualizations saved to: {output_path}")
    print("="*70)
    
    return results_no_ao, results_ao


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AO comparison visualizations")
    parser.add_argument("--cn2", type=float, default=10e-12,
                       help="Turbulence strength (m^-2/3), default: 10e-12")
    parser.add_argument("--n-zernike", type=int, default=35,
                       help="Number of Zernike modes, default: 35")
    parser.add_argument("--output-dir", type=str, default="plots/ao_comparison",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    generate_ao_comparison_visualizations(
        cn2=args.cn2,
        n_zernike=args.n_zernike,
        output_dir=args.output_dir
    )

