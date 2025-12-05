"""Verify AO correction is actually being applied and working."""

import numpy as np
import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(SCRIPT_DIR, "models"))

from adaptiveOptics import ModalController, decompose_phase_zernike, reconstruct_phase_zernike
from turbulence import apply_multi_layer_turbulence, create_multi_layer_screens
from lgBeam import LaguerreGaussianBeam

# Create test with known aberration
wavelength = 1550e-9
w0 = 25e-3
distance = 1000.0
N = 256
oversampling = 2

beam = LaguerreGaussianBeam(p=0, l=1, wavelength=wavelength, w0=w0)
beam_waist_L = beam.beam_waist(distance)
D = oversampling * 6.0 * beam_waist_L
delta = D / N

x = np.linspace(-D/2, D/2, N)
y = np.linspace(-D/2, D/2, N)
X, Y = np.meshgrid(x, y, indexing='ij')
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

# Create pristine field
E_pristine = beam.generate_beam_field(R, PHI, 0.0)

# Apply turbulence
print("Applying turbulence...")
layers = create_multi_layer_screens(
    distance, num_screens=10, wavelength=wavelength,
    ground_Cn2=5e-15, L0=10.0, l0=0.005,
    cn2_model="uniform", verbose=False
)

result = apply_multi_layer_turbulence(
    E_pristine, beam, layers, distance,
    N=N, oversampling=oversampling, L0=10.0, l0=0.005
)

E_distorted = result['final_field']
E_pristine_prop = result.get('pristine_field', E_pristine)

# Grid info
grid_info = {
    'x': x, 'y': y, 'X': X, 'Y': Y, 'R': R, 'PHI': PHI,
    'delta': delta, 'D': D, 'N': N,
    'pristine_field': E_pristine_prop,
    'phase_screens': result.get('phase_screens', None)
}

aperture_radius = np.max(R) * 0.8

# Apply AO
print("Applying AO correction...")
controller = ModalController(n_zernike_modes=15, aperture_radius=aperture_radius)
E_corrected = controller.estimate_and_correct(
    E_distorted, grid_info,
    reference_field=E_pristine_prop,
    phase_screens=result.get('phase_screens', None)
)

# Analyze results
phase_error_before = np.angle(E_distorted * np.conj(E_pristine_prop))
phase_error_after = np.angle(E_corrected * np.conj(E_pristine_prop))

mask = (R <= aperture_radius).astype(float)
rms_before = np.sqrt(np.mean((phase_error_before * mask)**2))
rms_after = np.sqrt(np.mean((phase_error_after * mask)**2))

# Strehl ratio
I_pristine = np.abs(E_pristine_prop)**2
I_distorted = np.abs(E_distorted)**2
I_corrected = np.abs(E_corrected)**2

strehl_before = np.max(I_distorted) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0
strehl_after = np.max(I_corrected) / np.max(I_pristine) if np.max(I_pristine) > 0 else 0

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"RMS Phase Error:")
print(f"  Before: {rms_before:.4f} rad ({np.degrees(rms_before):.2f}°)")
print(f"  After:  {rms_after:.4f} rad ({np.degrees(rms_after):.2f}°)")
print(f"  Reduction: {(1 - rms_after/rms_before)*100:.1f}%")

print(f"\nStrehl Ratio:")
print(f"  Before: {strehl_before:.4f}")
print(f"  After:  {strehl_after:.4f}")
print(f"  Improvement: {(strehl_after - strehl_before)*100:.2f}%")

# Show Zernike coefficients
if controller.coefficients_history:
    coeffs = controller.coefficients_history[-1]
    print(f"\nDominant Aberrations (Zernike coefficients):")
    sorted_coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    mode_names = {1: "Piston", 2: "Tip", 3: "Tilt", 4: "Defocus",
                  5: "Astig 45°", 6: "Astig 0°", 7: "Coma y", 8: "Coma x"}
    for j, a_j in sorted_coeffs[:8]:
        name = mode_names.get(j, f"Mode {j}")
        print(f"  {name:12s} (Z{j:2d}): {a_j:8.4f} rad")

if rms_after < rms_before * 0.9:
    print(f"\n✓ AO is working! Phase error reduced by {(1-rms_after/rms_before)*100:.1f}%")
else:
    print(f"\n⚠ AO correction minimal - may need stronger turbulence or different approach")

