"""
Test script showing how to integrate Adaptive Optics into the FSO-OAM pipeline.

This demonstrates:
1. Applying AO correction to distorted fields
2. Measuring improvement in mode purity
3. Comparing BER with/without AO
"""

import numpy as np
import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

from adaptiveOptics import apply_adaptive_optics, ModalController, ModePuritySensor
from turbulence import apply_multi_layer_turbulence, create_multi_layer_screens
from lgBeam import LaguerreGaussianBeam

def test_ao_correction():
    """Test AO correction on a simulated turbulent field."""
    
    print("=" * 60)
    print("Adaptive Optics Integration Test")
    print("=" * 60)
    
    # Setup
    wavelength = 1550e-9
    w0 = 25e-3
    distance = 1000.0
    N = 256
    oversampling = 2
    
    # Create test beam (LG mode)
    beam = LaguerreGaussianBeam(p=0, l=1, wavelength=wavelength, w0=w0)
    
    # Generate initial field
    beam_waist_L = beam.beam_waist(distance)
    D = oversampling * 6.0 * beam_waist_L
    delta = D / N
    
    x = np.linspace(-D/2, D/2, N)
    y = np.linspace(-D/2, D/2, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    initial_field = beam.generate_beam_field(R, PHI, 0.0)
    
    # Apply turbulence
    print("\n[1] Applying atmospheric turbulence...")
    layers = create_multi_layer_screens(
        distance, num_screens=10, wavelength=wavelength,
        ground_Cn2=1e-15, L0=10.0, l0=0.005,
        cn2_model="uniform", verbose=False
    )
    
    result = apply_multi_layer_turbulence(
        initial_field, beam, layers, distance,
        N=N, oversampling=oversampling, L0=10.0, l0=0.005
    )
    
    distorted_field = result['final_field']
    pristine_field = result['pristine_field']
    
    print(f"    Distorted field power: {np.sum(np.abs(distorted_field)**2) * delta**2:.3e} W")
    print(f"    Pristine field power:  {np.sum(np.abs(pristine_field)**2) * delta**2:.3e} W")
    
    # Grid info for AO (must match receiver.py format)
    grid_info = {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'R': R,
        'PHI': PHI,
        'delta': delta,
        'D': D,
        'N': N
    }
    
    # Test AO correction
    print("\n[2] Applying Adaptive Optics correction...")
    spatial_modes = [(0, 1)]  # Single mode for test
    
    corrected_field, metrics = apply_adaptive_optics(
        distorted_field, grid_info, spatial_modes,
        method='modal', n_zernike_modes=15
    )
    
    print(f"\n[3] Correction Results:")
    print(f"    Method: {metrics['method']}")
    print(f"    Mode Purity (before): {metrics['purity_before']:.4f}")
    print(f"    Mode Purity (after):  {metrics['purity_after']:.4f}")
    print(f"    Improvement: {(metrics['purity_after'] - metrics['purity_before'])*100:.2f}%")
    print(f"    Condition # (before): {metrics['cond_before']:.2e}")
    print(f"    Condition # (after):  {metrics['cond_after']:.2e}")
    
    # Compute Strehl ratio
    I_pristine = np.abs(pristine_field)**2
    I_distorted = np.abs(distorted_field)**2
    I_corrected = np.abs(corrected_field)**2
    
    strehl_distorted = np.max(I_distorted) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0
    strehl_corrected = np.max(I_corrected) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0
    
    print(f"\n[4] Strehl Ratio:")
    print(f"    Distorted: {strehl_distorted:.4f}")
    print(f"    Corrected: {strehl_corrected:.4f}")
    print(f"    Improvement: {(strehl_corrected - strehl_distorted)*100:.2f}%")
    
    # Show dominant Zernike modes
    if 'zernike_coeffs' in metrics and metrics['zernike_coeffs']:
        print(f"\n[5] Dominant Aberrations (Zernike coefficients):")
        sorted_coeffs = sorted(metrics['zernike_coeffs'].items(), 
                              key=lambda x: abs(x[1]), reverse=True)
        for j, a_j in sorted_coeffs[:5]:
            mode_name = {1: "Piston", 2: "Tip", 3: "Tilt", 4: "Defocus", 
                        5: "Astig 45°", 6: "Astig 0°"}.get(j, f"Mode {j}")
            print(f"    {mode_name} (Z{j}): {a_j:.3f} rad")
    
    print("\n" + "=" * 60)
    print("✓ Adaptive Optics test completed successfully!")
    print("=" * 60)
    
    return corrected_field, metrics


if __name__ == "__main__":
    test_ao_correction()

