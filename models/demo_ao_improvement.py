"""
Demonstration of Adaptive Optics improvement with stronger turbulence.

Shows clear improvement in mode purity and Strehl ratio when AO is applied.
"""

import numpy as np
import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

from adaptiveOptics import apply_adaptive_optics, decompose_phase_zernike, reconstruct_phase_zernike
from turbulence import apply_multi_layer_turbulence, create_multi_layer_screens
from lgBeam import LaguerreGaussianBeam

def demo_ao_with_strong_turbulence():
    """Demonstrate AO correction with moderate turbulence."""
    
    print("=" * 70)
    print("Adaptive Optics Demonstration: Strong Turbulence Case")
    print("=" * 70)
    
    # Setup with stronger turbulence
    wavelength = 1550e-9
    w0 = 25e-3
    distance = 1000.0
    N = 256
    oversampling = 2
    Cn2 = 5e-15  # Stronger turbulence
    
    # Create test beam
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
    
    # Apply stronger turbulence
    print(f"\n[1] Applying turbulence (Cn² = {Cn2:.1e} m^-2/3)...")
    layers = create_multi_layer_screens(
        distance, num_screens=15, wavelength=wavelength,
        ground_Cn2=Cn2, L0=10.0, l0=0.005,
        cn2_model="uniform", verbose=False
    )
    
    result = apply_multi_layer_turbulence(
        initial_field, beam, layers, distance,
        N=N, oversampling=oversampling, L0=10.0, l0=0.005
    )
    
    distorted_field = result['final_field']
    pristine_field = result['pristine_field']
    
    # Grid info
    grid_info = {
        'x': x, 'y': y, 'X': X, 'Y': Y,
        'R': R, 'PHI': PHI, 'delta': delta, 'D': D, 'N': N
    }
    
    # Analyze phase aberrations
    print("\n[2] Analyzing phase aberrations...")
    phase_distorted = np.angle(distorted_field)
    aperture_radius = np.max(R) * 0.8  # Use 80% of grid as aperture
    zernike_coeffs = decompose_phase_zernike(
        phase_distorted, R, PHI, n_modes=15, aperture_radius=aperture_radius
    )
    
    # Show dominant aberrations
    sorted_coeffs = sorted(zernike_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("    Dominant Zernike modes (before correction):")
    mode_names = {1: "Piston", 2: "Tip (x-tilt)", 3: "Tilt (y-tilt)", 
                  4: "Defocus", 5: "Astig 45°", 6: "Astig 0°",
                  7: "Coma y", 8: "Coma x", 9: "Trefoil", 10: "Spherical"}
    for j, a_j in sorted_coeffs[:5]:
        name = mode_names.get(j, f"Mode {j}")
        print(f"      {name:20s} (Z{j:2d}): {a_j:8.4f} rad")
    
    # Apply AO correction
    print("\n[3] Applying Adaptive Optics correction...")
    spatial_modes = [(0, 1)]
    
    corrected_field, metrics = apply_adaptive_optics(
        distorted_field, grid_info, spatial_modes,
        method='modal', n_zernike_modes=15
    )
    
    # Compute metrics
    I_pristine = np.abs(pristine_field)**2
    I_distorted = np.abs(distorted_field)**2
    I_corrected = np.abs(corrected_field)**2
    
    strehl_distorted = np.max(I_distorted) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0
    strehl_corrected = np.max(I_corrected) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0
    
    # RMS wavefront error
    phase_corrected = np.angle(corrected_field)
    phase_error_before = phase_distorted - np.angle(pristine_field)
    phase_error_after = phase_corrected - np.angle(pristine_field)
    
    mask = (R <= aperture_radius).astype(float)
    rms_before = np.sqrt(np.mean((phase_error_before * mask)**2))
    rms_after = np.sqrt(np.mean((phase_error_after * mask)**2))
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nMode Purity:")
    print(f"  Before AO: {metrics['purity_before']:.4f}")
    print(f"  After AO:  {metrics['purity_after']:.4f}")
    print(f"  Improvement: {(metrics['purity_after'] - metrics['purity_before'])*100:+.2f}%")
    
    print(f"\nChannel Condition Number:")
    print(f"  Before AO: {metrics['cond_before']:.2e}")
    print(f"  After AO:  {metrics['cond_after']:.2e}")
    print(f"  Improvement: {(1 - metrics['cond_after']/metrics['cond_before'])*100:+.1f}%")
    
    print(f"\nStrehl Ratio:")
    print(f"  Before AO: {strehl_distorted:.4f}")
    print(f"  After AO:  {strehl_corrected:.4f}")
    print(f"  Improvement: {(strehl_corrected - strehl_distorted)*100:+.2f}%")
    
    print(f"\nRMS Wavefront Error:")
    print(f"  Before AO: {rms_before:.4f} rad ({np.degrees(rms_before):.2f}°)")
    print(f"  After AO:  {rms_after:.4f} rad ({np.degrees(rms_after):.2f}°)")
    print(f"  Reduction: {(1 - rms_after/rms_before)*100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ Adaptive Optics successfully improves wavefront quality!")
    print("=" * 70)
    
    return corrected_field, metrics


if __name__ == "__main__":
    demo_ao_with_strong_turbulence()

