"""
Adaptive Optics Simulation for FSO-OAM Communication Systems

This module provides:
1. Zernike polynomial generation and phase decomposition
2. Wavefront sensing (mode purity, phase retrieval)
3. Correction devices (SLM, Deformable Mirror models)
4. Control algorithms (modal control, iterative optimization)
5. Integration with existing turbulence model

Literature References:
- Noll (1976): Zernike polynomials for atmospheric turbulence
- Hardy (1998): Adaptive Optics for Astronomical Telescopes
- Booth (2014): Wavefront sensorless adaptive optics
"""

import numpy as np
from scipy.special import factorial, jv
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import warnings
from typing import Dict, Tuple, List, Optional, Callable
import sys
import os

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

warnings.filterwarnings('ignore')


# ============================================================================
# ZERNIKE POLYNOMIALS
# ============================================================================

def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute radial part of Zernike polynomial R_n^m(ρ).
    
    Formula: R_n^m(ρ) = Σ_{k=0}^{(n-|m|)/2} (-1)^k * (n-k)! / 
                        [k! * ((n+|m|)/2 - k)! * ((n-|m|)/2 - k)!] * ρ^(n-2k)
    
    Args:
        n: Radial order (n ≥ 0)
        m: Azimuthal order (|m| ≤ n, n-|m| even)
        rho: Normalized radial coordinate (0 ≤ ρ ≤ 1)
    
    Returns:
        R_n^m(ρ): Radial polynomial values
    """
    rho = np.asarray(rho)
    if n < 0 or abs(m) > n or (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    
    if n == 0 and m == 0:
        return np.ones_like(rho)
    
    result = np.zeros_like(rho)
    k_max = (n - abs(m)) // 2
    
    for k in range(k_max + 1):
        numerator = factorial(n - k)
        denom1 = factorial(k)
        denom2 = factorial((n + abs(m)) // 2 - k)
        denom3 = factorial((n - abs(m)) // 2 - k)
        
        coeff = ((-1) ** k) * numerator / (denom1 * denom2 * denom3)
        power = n - 2 * k
        result += coeff * (rho ** power)
    
    return result


def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute full Zernike polynomial Z_n^m(ρ,θ).
    
    Z_n^m(ρ,θ) = R_n^m(ρ) * {
        √2 * cos(mθ)  if m > 0
        √2 * sin(|m|θ) if m < 0
        1             if m = 0
    }
    
    Noll indexing: j = (n(n+1)/2) + |m| + (m>0) - 1
    Common modes:
        j=1: Piston (n=0, m=0)
        j=2: Tip (n=1, m=-1)
        j=3: Tilt (n=1, m=1)
        j=4: Defocus (n=2, m=0)
        j=5: Astigmatism 45° (n=2, m=-2)
        j=6: Astigmatism 0° (n=2, m=2)
    
    Args:
        n: Radial order
        m: Azimuthal order
        rho: Normalized radial coordinate (0 ≤ ρ ≤ 1)
        theta: Azimuthal angle (0 ≤ θ < 2π)
    
    Returns:
        Z_n^m(ρ,θ): Zernike polynomial values
    """
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    
    # Radial part
    R = zernike_radial(n, abs(m), rho)
    
    # Azimuthal part
    if m > 0:
        azimuthal = np.sqrt(2) * np.cos(m * theta)
    elif m < 0:
        azimuthal = np.sqrt(2) * np.sin(abs(m) * theta)
    else:  # m == 0
        azimuthal = np.ones_like(theta)
    
    # Combine
    zernike = R * azimuthal
    
    # Apply aperture mask (zero outside unit circle)
    mask = (rho <= 1.0).astype(float)
    return zernike * mask


def noll_to_zernike(j: int) -> Tuple[int, int]:
    """
    Convert Noll index j to (n, m) Zernike indices.
    
    Noll ordering (OSA/ANSI standard):
        j=1: (0,0) Piston
        j=2: (1,-1) Tip (x-tilt)
        j=3: (1,1) Tilt (y-tilt)
        j=4: (2,0) Defocus
        j=5: (2,-2) Astigmatism 45°
        j=6: (2,2) Astigmatism 0°
        j=7: (3,-1) Coma y
        j=8: (3,1) Coma x
        ...
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    
    j -= 1  # Convert to 0-based
    
    n = 0
    while j > n:
        n += 1
        j -= n + 1
    
    m = -n + 2 * j
    return (n, m)


def zernike_to_noll(n: int, m: int) -> int:
    """Convert (n, m) to Noll index j."""
    if n < 0 or abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"Invalid Zernike indices: n={n}, m={m}")
    
    j = n * (n + 1) // 2 + abs(m)
    if m > 0 and n != 0:
        j += 1
    elif m < 0:
        j += 1
    
    return j + 1  # Convert to 1-based


def generate_zernike_basis(n_modes: int, rho: np.ndarray, theta: np.ndarray, 
                          aperture_radius: float = 1.0) -> Dict[int, np.ndarray]:
    """
    Generate Zernike polynomial basis set.
    
    Args:
        n_modes: Number of Zernike modes to generate (starting from j=1)
        rho: Radial coordinates (normalized to [0, 1] if aperture_radius=1.0)
        theta: Azimuthal angles
        aperture_radius: Aperture radius for normalization
    
    Returns:
        Dictionary mapping Noll index j to Zernike polynomial array
    """
    rho_norm = rho / aperture_radius if aperture_radius > 0 else rho
    zernike_basis = {}
    
    for j in range(1, n_modes + 1):
        n, m = noll_to_zernike(j)
        zernike_basis[j] = zernike_polynomial(n, m, rho_norm, theta)
    
    return zernike_basis


def decompose_phase_zernike(phase: np.ndarray, rho: np.ndarray, theta: np.ndarray,
                            n_modes: int = 15, aperture_radius: float = 1.0,
                            grid_info: Optional[Dict] = None) -> Dict[int, float]:
    """
    Decompose phase screen into Zernike coefficients using LEAST SQUARES.
    
    OUTSIDE-THE-BOX APPROACH: Instead of integration (sensitive to grid/mask issues),
    we fit phase = Σ a_j * Z_j using least squares. This is more robust and accurate.
    
    Method: Solve min ||phase - Z*a||² → a = (Z^T * Z)^(-1) * Z^T * phase
    
    Args:
        phase: Phase screen [rad]
        rho: Radial coordinates
        theta: Azimuthal angles
        n_modes: Number of Zernike modes to use
        aperture_radius: Aperture radius
        grid_info: Grid information (optional, for delta)
    
    Returns:
        Dictionary mapping Noll index j to coefficient a_j
    """
    phase = np.asarray(phase)
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    
    # Normalize coordinates
    rho_norm = rho / aperture_radius if aperture_radius > 0 else rho
    mask = (rho_norm <= 1.0).astype(float)
    
    # OUTSIDE-THE-BOX: Use least squares fitting instead of integration!
    # This avoids all the integration area/normalization issues
    
    # Flatten arrays for matrix operations
    phase_flat = phase.flatten()
    mask_flat = mask.flatten()
    
    # Only use points inside aperture
    valid_idx = mask_flat > 0.5
    phase_valid = phase_flat[valid_idx]
    n_valid = np.sum(valid_idx)
    
    if n_valid == 0:
        return {j: 0.0 for j in range(1, n_modes + 1)}
    
    # Generate Zernike basis and build design matrix
    zernike_basis = generate_zernike_basis(n_modes, rho, theta, aperture_radius)
    mode_indices = sorted(zernike_basis.keys())
    
    # Build design matrix Z: each column is a flattened Zernike mode
    Z_matrix = np.zeros((n_valid, len(mode_indices)))
    for col_idx, j in enumerate(mode_indices):
        z_j_flat = zernike_basis[j].flatten()
        Z_matrix[:, col_idx] = z_j_flat[valid_idx]
    
    # Least squares: min ||phase - Z*a||²
    # Solution: a = (Z^T * Z)^(-1) * Z^T * phase
    try:
        # Normal equations
        ZTZ = Z_matrix.T @ Z_matrix
        ZT_phase = Z_matrix.T @ phase_valid
        a_vec = np.linalg.solve(ZTZ, ZT_phase)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse for numerical stability
        a_vec = np.linalg.pinv(Z_matrix) @ phase_valid
    
    # Convert to dictionary
    coefficients = {j: float(a_vec[col_idx]) 
                   for col_idx, j in enumerate(mode_indices)}
    
    return coefficients


def reconstruct_phase_zernike(coefficients: Dict[int, float], rho: np.ndarray, 
                              theta: np.ndarray, aperture_radius: float = 1.0) -> np.ndarray:
    """
    Reconstruct phase from Zernike coefficients.
    
    φ(ρ,θ) = Σ_j a_j * Z_j(ρ,θ)
    
    Args:
        coefficients: Dictionary of {Noll_index: coefficient}
        rho: Radial coordinates
        theta: Azimuthal angles
        aperture_radius: Aperture radius
    
    Returns:
        Reconstructed phase screen
    """
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    phase = np.zeros_like(rho, dtype=float)
    
    for j, a_j in coefficients.items():
        n, m = noll_to_zernike(j)
        rho_norm = rho / aperture_radius if aperture_radius > 0 else rho
        z_j = zernike_polynomial(n, m, rho_norm, theta)
        phase += a_j * z_j
    
    return phase


# ============================================================================
# WAVEFRONT SENSING
# ============================================================================

class ModePuritySensor:
    """
    Wavefront sensor using OAM mode demultiplexing output.
    
    Measures mode crosstalk matrix H to assess wavefront quality.
    High off-diagonal elements indicate need for correction.
    """
    
    def __init__(self, spatial_modes: List[Tuple[int, int]], wavelength: float, 
                 w0: float, z_distance: float):
        self.spatial_modes = spatial_modes
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
    
    def measure_crosstalk(self, received_field: np.ndarray, grid_info: Dict,
                         reference_modes: Optional[Dict] = None) -> np.ndarray:
        """
        Measure full crosstalk matrix by projecting received field onto all reference modes.
        
        For a multiplexed field, we compute the projection onto each reference mode.
        This gives us the full crosstalk matrix H where:
        - H[i,j] = projection of received field onto mode i when mode j is reference
        - Diagonal H[i,i] = power in correct mode i
        - Off-diagonal H[i,j] = crosstalk from mode j into mode i
        
        Returns:
            Full crosstalk matrix H (n_modes x n_modes)
        """
        from receiver import OAMDemultiplexer
        
        demux = OAMDemultiplexer(self.spatial_modes, self.wavelength, 
                                 self.w0, self.z_distance)
        
        X = grid_info['X']
        Y = grid_info['Y']
        delta = grid_info['delta']
        dA = delta ** 2
        
        n_modes = len(self.spatial_modes)
        H = np.zeros((n_modes, n_modes), dtype=complex)
        
        # Generate all reference fields first
        ref_fields = {}
        for i, mode_key in enumerate(self.spatial_modes):
            ref = demux.reference_field(mode_key, X, Y, delta, 
                                       grid_z=self.z_distance,
                                       tx_beam_obj=None, scaling_factor=None)
            
            # Normalize reference fields for proper projection
            ref_energy = np.sum(np.abs(ref) ** 2) * dA
            if ref_energy > 1e-20:
                ref_normalized = ref / np.sqrt(ref_energy)
            else:
                ref_normalized = ref
            
            ref_fields[mode_key] = ref_normalized
        
        # Compute full crosstalk matrix
        # For a multiplexed field, true crosstalk is difficult to measure without TX info
        # Instead, we measure "mode coupling" which is a proxy for crosstalk
        
        # Key insight: With turbulence, modes couple. We can measure this by:
        # 1. Computing how well the received field projects onto each reference mode (diagonal)
        # 2. Measuring how much the received field's structure deviates from ideal modes (off-diagonal)
        
        # Compute diagonal elements: projection onto each normalized reference
        for i, mode_i in enumerate(self.spatial_modes):
            ref_i = ref_fields[mode_i]
            projection = np.sum(received_field * np.conj(ref_i)) * dA
            H[i, i] = projection
        
        # For off-diagonal: measure mode coupling by computing cross-correlations
        # The idea: if modes are perfectly orthogonal, cross-projections are zero
        # With turbulence, the received field has mixed mode content, so cross-projections increase
        
        # Method: Compute the "mode mixing" by measuring how much each mode's
        # reference field appears in the received field when projected onto other modes
        
        # Get unnormalized reference fields to compute actual mode structure
        ref_fields_unnorm = {}
        ref_energies = {}
        for i, mode_i in enumerate(self.spatial_modes):
            ref_unnorm = demux.reference_field(mode_i, X, Y, delta, 
                                               grid_z=self.z_distance,
                                               tx_beam_obj=None, scaling_factor=None)
            ref_fields_unnorm[mode_i] = ref_unnorm
            ref_energies[mode_i] = np.sum(np.abs(ref_unnorm) ** 2) * dA
        
        # Compute off-diagonal elements: measure mode coupling
        # H[i,j] represents how much mode j's structure appears in mode i's projection
        for i, mode_i in enumerate(self.spatial_modes):
            ref_i_norm = ref_fields[mode_i]
            
            for j, mode_j in enumerate(self.spatial_modes):
                if i != j:
                    ref_j_unnorm = ref_fields_unnorm[mode_j]
                    
                    # Method: Compute how much of mode j's unnormalized structure
                    # projects onto mode i's normalized reference
                    # This measures mode coupling/crosstalk
                    
                    # The received field has contributions from all modes
                    # We estimate crosstalk by computing the correlation between
                    # the received field's structure and mode j, when projected onto mode i
                    
                    # Compute: H[i,j] = <E_rx | ref_i> * <E_rx | ref_j*> / <ref_j | ref_j>
                    # This captures how mode j's power couples into mode i
                    
                    # Better approach: Measure the "mode mixing coefficient"
                    # by computing how much the received field, when filtered by mode j's structure,
                    # projects onto mode i
                    
                    # Filter received field by mode j's structure (weight by mode j's amplitude)
                    mode_j_weight = np.abs(ref_j_unnorm) / (np.sqrt(ref_energies[mode_j]) + 1e-20)
                    filtered_field = received_field * mode_j_weight
                    
                    # Project filtered field onto mode i
                    crosstalk_ij = np.sum(filtered_field * np.conj(ref_i_norm)) * dA
                    
                    # Scale by the power in mode j to get relative crosstalk
                    power_j = np.abs(H[j, j])
                    if power_j > 1e-20:
                        # Normalize crosstalk relative to mode j's power
                        H[i, j] = crosstalk_ij * (power_j / (np.abs(crosstalk_ij) + 1e-20))
                    else:
                        H[i, j] = crosstalk_ij
        
        return H
    
    def compute_purity_metric(self, H: np.ndarray) -> float:
        """
        Compute mode purity metric.
        
        Purity = trace(|H|²) / sum(|H|²)
        Higher is better (1.0 = perfect separation)
        """
        H_abs_sq = np.abs(H) ** 2
        trace = np.trace(H_abs_sq)
        total = np.sum(H_abs_sq)
        return trace / total if total > 0 else 0.0
    
    def compute_condition_metric(self, H: np.ndarray) -> float:
        """
        Compute channel condition number.
        
        cond(H) = σ_max / σ_min
        Lower is better (1.0 = perfect)
        """
        try:
            return np.linalg.cond(H)
        except:
            return np.inf
    
    def compute_coupling_efficiency(self, H: np.ndarray) -> float:
        """
        Compute coupling efficiency: average power in correct mode.
        
        CE = (1/N) * Σ_i |H_ii|² / Σ_i,j |H_ij|²
        Higher is better (1.0 = perfect)
        """
        H_abs_sq = np.abs(H) ** 2
        diagonal_power = np.sum(np.diag(H_abs_sq))
        total_power = np.sum(H_abs_sq)
        n_modes = H.shape[0]
        return (diagonal_power / n_modes) / (total_power / (n_modes * n_modes)) if total_power > 0 else 0.0
    
    def compute_crosstalk_dB(self, H: np.ndarray) -> float:
        """
        Compute average crosstalk in dB.
        
        XT = 10*log10(off_diagonal_power / diagonal_power)
        Lower is better (negative dB means less crosstalk)
        
        This measures the ratio of power in off-diagonal elements (crosstalk)
        to power in diagonal elements (desired signal).
        """
        H_abs_sq = np.abs(H) ** 2
        diagonal_power = np.sum(np.diag(H_abs_sq))
        off_diagonal_power = np.sum(H_abs_sq) - diagonal_power
        
        if diagonal_power > 0 and off_diagonal_power > 0:
            ratio = off_diagonal_power / diagonal_power
            return 10 * np.log10(ratio)
        elif diagonal_power > 0:
            # No crosstalk (perfect case)
            return -np.inf
        else:
            # No signal
            return np.inf


class PhaseRetrievalSensor:
    """
    Phase retrieval from intensity measurements (phase diversity).
    
    Uses iterative algorithms (Gerchberg-Saxton, hybrid input-output)
    to reconstruct phase from intensity patterns.
    """
    
    def __init__(self, wavelength: float, grid_info: Dict):
        self.wavelength = wavelength
        self.grid_info = grid_info
    
    def gerchberg_saxton(self, intensity_focused: np.ndarray, 
                        intensity_defocused: np.ndarray,
                        defocus_distance: float, max_iterations: int = 50) -> np.ndarray:
        """
        Gerchberg-Saxton algorithm for phase retrieval.
        
        Iteratively propagates between focused and defocused planes,
        enforcing intensity constraints.
        """
        from turbulence import angular_spectrum_propagation
        
        delta = self.grid_info['delta']
        N = intensity_focused.shape[0]
        
        # Initialize: random phase
        phase = np.random.rand(N, N) * 2 * np.pi - np.pi
        field_focused = np.sqrt(intensity_focused) * np.exp(1j * phase)
        
        for iteration in range(max_iterations):
            # Propagate to defocused plane
            field_defocused = angular_spectrum_propagation(
                field_focused, delta, self.wavelength, defocus_distance
            )
            
            # Enforce defocused intensity constraint
            amplitude_defocused = np.sqrt(intensity_defocused)
            field_defocused = amplitude_defocused * np.exp(1j * np.angle(field_defocused))
            
            # Propagate back to focused plane
            field_focused = angular_spectrum_propagation(
                field_defocused, delta, self.wavelength, -defocus_distance
            )
            
            # Enforce focused intensity constraint
            amplitude_focused = np.sqrt(intensity_focused)
            phase = np.angle(field_focused)
            field_focused = amplitude_focused * np.exp(1j * phase)
        
        return phase


# ============================================================================
# CORRECTION DEVICES
# ============================================================================

class SpatialLightModulator:
    """
    Model of a Spatial Light Modulator (SLM) for phase correction.
    
    Pixelated phase modulator with configurable resolution and phase range.
    """
    
    def __init__(self, n_pixels_x: int, n_pixels_y: int, pixel_pitch: float,
                 phase_range: Tuple[float, float] = (0.0, 2*np.pi),
                 quantization_bits: int = 8, wavelength: float = 1550e-9):
        self.n_pixels_x = n_pixels_x
        self.n_pixels_y = n_pixels_y
        self.pixel_pitch = pixel_pitch
        self.phase_range = phase_range
        self.quantization_bits = quantization_bits
        self.wavelength = wavelength
        
        # Phase map (in radians)
        self.phase_map = np.zeros((n_pixels_y, n_pixels_x), dtype=float)
        
        # Quantization levels
        self.phase_step = (phase_range[1] - phase_range[0]) / (2 ** quantization_bits)
    
    def set_phase_map(self, phase_map: np.ndarray, smooth: bool = False):
        """
        Set phase correction pattern.
        
        Args:
            phase_map: Desired phase pattern [rad] (will be resampled to SLM grid)
            smooth: Apply Gaussian smoothing to reduce pixelation effects
        """
        # Resample to SLM grid if needed
        if phase_map.shape != (self.n_pixels_y, self.n_pixels_x):
            from scipy.ndimage import zoom
            zoom_y = self.n_pixels_y / phase_map.shape[0]
            zoom_x = self.n_pixels_x / phase_map.shape[1]
            phase_map = zoom(phase_map, (zoom_y, zoom_x), order=1)
        
        # Smooth if requested
        if smooth:
            phase_map = gaussian_filter(phase_map, sigma=0.5)
        
        # Wrap to phase range
        phase_min, phase_max = self.phase_range
        self.phase_map = np.mod(phase_map - phase_min, phase_max - phase_min) + phase_min
        
        # Quantize
        if self.quantization_bits < 16:
            self.phase_map = np.round(self.phase_map / self.phase_step) * self.phase_step
    
    def apply_correction(self, field: np.ndarray, grid_info: Dict) -> np.ndarray:
        """
        Apply phase correction to field.
        
        Args:
            field: Input field E(x,y)
            grid_info: Grid information (for resampling phase map)
        
        Returns:
            Corrected field: E_corrected = E * exp(-i * φ_corr)
        """
        # Resample phase map to field grid
        if field.shape != self.phase_map.shape:
            from scipy.ndimage import zoom
            zoom_y = field.shape[0] / self.n_pixels_y
            zoom_x = field.shape[1] / self.n_pixels_x
            phase_resampled = zoom(self.phase_map, (zoom_y, zoom_x), order=1)
        else:
            phase_resampled = self.phase_map
        
        # Apply correction (negative to cancel aberrations)
        return field * np.exp(-1j * phase_resampled)
    
    def set_zernike_correction(self, coefficients: Dict[int, float], 
                               grid_info: Dict, aperture_radius: float):
        """
        Set correction from Zernike coefficients.
        
        Convenience method to generate phase map from modal correction.
        """
        rho = grid_info['R']
        theta = grid_info['PHI']
        phase_corr = reconstruct_phase_zernike(coefficients, rho, theta, aperture_radius)
        self.set_phase_map(phase_corr, smooth=True)


class DeformableMirror:
    """
    Model of a Deformable Mirror (DM) with actuator influence functions.
    
    More realistic than SLM but requires influence function modeling.
    """
    
    def __init__(self, n_actuators_x: int, n_actuators_y: int, 
                 actuator_spacing: float, influence_type: str = 'gaussian',
                 coupling: float = 0.1, wavelength: float = 1550e-9):
        self.n_actuators_x = n_actuators_x
        self.n_actuators_y = n_actuators_y
        self.actuator_spacing = actuator_spacing
        self.influence_type = influence_type
        self.coupling = coupling
        self.wavelength = wavelength
        
        # Actuator commands (surface displacement in meters)
        self.actuator_commands = np.zeros((n_actuators_y, n_actuators_x))
        
        # Precompute influence matrix
        self._compute_influence_matrix()
    
    def _compute_influence_matrix(self):
        """Precompute influence functions for each actuator."""
        # This would compute the influence function h(x,y) for each actuator
        # For now, simplified model
        pass
    
    def set_actuator_commands(self, commands: np.ndarray):
        """Set actuator surface displacements [m]."""
        self.actuator_commands = np.asarray(commands)
    
    def apply_correction(self, field: np.ndarray, grid_info: Dict) -> np.ndarray:
        """
        Apply DM correction to field.
        
        Phase correction: φ = (4π/λ) * surface_displacement
        """
        # Compute surface displacement at field grid points
        # (would use influence functions in full implementation)
        surface = self.actuator_commands  # Simplified
        
        # Convert to phase
        phase_corr = (4 * np.pi / self.wavelength) * surface
        
        # Apply correction
        return field * np.exp(-1j * phase_corr)


# ============================================================================
# CONTROL ALGORITHMS
# ============================================================================

class ModalController:
    """
    Modal control using Zernike decomposition.
    
    Estimates phase aberrations, decomposes into Zernike modes,
    and applies correction.
    """
    
    def __init__(self, n_zernike_modes: int = 15, aperture_radius: float = 1.0,
                 correction_device: Optional[SpatialLightModulator] = None):
        self.n_zernike_modes = n_zernike_modes
        self.aperture_radius = aperture_radius
        self.correction_device = correction_device
        self.coefficients_history = []
    
    def estimate_and_correct(self, received_field: np.ndarray, grid_info: Dict,
                            reference_field: Optional[np.ndarray] = None,
                            phase_screens: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Estimate phase aberrations and apply correction.
        
        FIXED: Properly extracts aberration phase by comparing received to pristine,
        or by using phase screens directly if available.
        
        Steps:
        1. Get aberration phase (from phase screens OR phase difference)
        2. Decompose into Zernike modes (skip piston)
        3. Generate correction phase (negative of aberration)
        4. Apply correction
        """
        rho = grid_info['R']
        theta = grid_info['PHI']
        
        # BEST METHOD: Use phase screens directly if available (most accurate)
        if phase_screens is not None and len(phase_screens) > 0:
            # Sum all phase screens to get total aberration
            # phase_screens is a list of 2D arrays
            phase_aberration = np.zeros_like(phase_screens[0])
            for screen in phase_screens:
                if screen.shape == phase_aberration.shape:
                    phase_aberration += screen
        # SECOND BEST: Compare to pristine reference field
        elif reference_field is not None:
            # Compute phase difference: aberration = arg(E_rx * conj(E_pristine))
            # This removes OAM phase and gives pure aberration
            E_rx_normalized = received_field / (np.abs(received_field) + 1e-12)
            E_ref_normalized = reference_field / (np.abs(reference_field) + 1e-12)
            phase_aberration = np.angle(E_rx_normalized * np.conj(E_ref_normalized))
            # Smooth to reduce noise
            phase_aberration = gaussian_filter(phase_aberration, sigma=1.5)
        else:
            # FALLBACK: Estimate from received field intensity-weighted phase
            # This is less accurate but better than nothing
            intensity = np.abs(received_field) ** 2
            phase_total = np.angle(received_field)
            
            # Remove piston and smooth
            mask = intensity > (np.max(intensity) * 0.1)
            if np.sum(mask) > 0:
                piston = np.mean(phase_total[mask])
                phase_aberration = phase_total - piston
            else:
                phase_aberration = phase_total - np.mean(phase_total)
            
            phase_aberration = gaussian_filter(phase_aberration, sigma=2.0)
        
        # Decompose aberration into Zernike modes
        coefficients = decompose_phase_zernike(
            phase_aberration, rho, theta, 
            n_modes=self.n_zernike_modes,
            aperture_radius=self.aperture_radius,
            grid_info=grid_info
        )
        
        # Remove piston (j=1) - it doesn't affect intensity/beam quality
        if 1 in coefficients:
            coefficients.pop(1)
        
        # Store history
        self.coefficients_history.append(coefficients.copy())
        
        # Generate correction (negative to cancel aberrations)
        correction_coeffs = {j: -a_j for j, a_j in coefficients.items()}
        phase_correction = reconstruct_phase_zernike(
            correction_coeffs, rho, theta, self.aperture_radius
        )
        
        # Apply correction
        if self.correction_device is not None:
            self.correction_device.set_phase_map(phase_correction, smooth=True)
            corrected_field = self.correction_device.apply_correction(received_field, grid_info)
        else:
            # Direct application
            corrected_field = received_field * np.exp(-1j * phase_correction)
        
        return corrected_field


class SensorlessOptimizer:
    """
    Sensorless adaptive optics using iterative optimization.
    
    Optimizes correction based on performance metric (mode purity, BER, etc.)
    without explicit wavefront sensing.
    """
    
    def __init__(self, spatial_modes: List[Tuple[int, int]], grid_info: Dict,
                 metric: str = 'mode_purity', correction_device: Optional[SpatialLightModulator] = None):
        self.spatial_modes = spatial_modes
        self.grid_info = grid_info
        self.metric = metric
        self.correction_device = correction_device
        self.sensor = ModePuritySensor(spatial_modes, 1550e-9, 25e-3, 1000.0)
    
    def optimize_correction(self, received_field: np.ndarray, 
                          max_iterations: int = 20) -> np.ndarray:
        """
        Optimize correction using coordinate descent.
        
        Optimizes one Zernike mode at a time.
        """
        rho = self.grid_info['R']
        theta = self.grid_info['PHI']
        aperture_radius = np.max(rho)
        
        # Start with low-order modes (tip, tilt, defocus, astigmatism)
        modes_to_optimize = [2, 3, 4, 5, 6]  # Noll indices
        
        best_field = received_field.copy()
        best_metric = self._compute_metric(best_field)
        
        for mode_j in modes_to_optimize:
            # Optimize this mode's coefficient
            def objective(coeff):
                # Generate correction for this mode only
                correction_coeffs = {mode_j: -coeff}
                phase_corr = reconstruct_phase_zernike(
                    correction_coeffs, rho, theta, aperture_radius
                )
                corrected = received_field * np.exp(-1j * phase_corr)
                metric = self._compute_metric(corrected)
                return -metric  # Minimize negative metric = maximize metric
            
            # Optimize
            result = minimize(objective, x0=0.0, method='BFGS', options={'maxiter': 10})
            if result.success:
                # Apply best correction for this mode
                correction_coeffs = {mode_j: -result.x[0]}
                phase_corr = reconstruct_phase_zernike(
                    correction_coeffs, rho, theta, aperture_radius
                )
                best_field = best_field * np.exp(-1j * phase_corr)
                best_metric = self._compute_metric(best_field)
        
        return best_field
    
    def _compute_metric(self, field: np.ndarray) -> float:
        """Compute performance metric."""
        if self.metric == 'mode_purity':
            H = self.sensor.measure_crosstalk(field, self.grid_info)
            return self.sensor.compute_purity_metric(H)
        elif self.metric == 'condition':
            H = self.sensor.measure_crosstalk(field, self.grid_info)
            return 1.0 / self.sensor.compute_condition_metric(H)  # Invert (lower is better)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


# ============================================================================
# INTEGRATION WITH PIPELINE
# ============================================================================

def apply_adaptive_optics(received_field: np.ndarray, grid_info: Dict,
                          spatial_modes: List[Tuple[int, int]],
                          method: str = 'modal',
                          n_zernike_modes: int = 15) -> Tuple[np.ndarray, Dict]:
    """
    Apply adaptive optics correction to received field.
    
    Convenience function for integration into pipeline.
    
    Args:
        received_field: Distorted received field E_rx(x,y)
        grid_info: Grid information dictionary
        spatial_modes: List of (p,l) mode tuples
        method: Correction method ('modal', 'sensorless', 'none')
        n_zernike_modes: Number of Zernike modes for modal correction
    
    Returns:
        corrected_field: Corrected field
        metrics: Dictionary with correction metrics
    """
    if method == 'none':
        return received_field, {'applied': False}
    
    aperture_radius = np.max(grid_info['R'])
    
    if method == 'modal':
        # Modal control
        # Get pristine reference and phase screens for accurate aberration estimation
        reference_field = grid_info.get('pristine_field', None)
        phase_screens = grid_info.get('phase_screens', None)
        
        controller = ModalController(
            n_zernike_modes=n_zernike_modes,
            aperture_radius=aperture_radius
        )
        corrected_field = controller.estimate_and_correct(
            received_field, grid_info, 
            reference_field=reference_field,
            phase_screens=phase_screens
        )
        
        # Compute comprehensive metrics
        sensor = ModePuritySensor(spatial_modes, 1550e-9, 25e-3, 1000.0)
        H_before = sensor.measure_crosstalk(received_field, grid_info)
        H_after = sensor.measure_crosstalk(corrected_field, grid_info)
        
        # Power metrics
        delta = grid_info.get('delta', 1.0)
        dA = delta ** 2
        power_before = np.sum(np.abs(received_field) ** 2) * dA
        power_after = np.sum(np.abs(corrected_field) ** 2) * dA
        power_loss_dB = 10 * np.log10(power_after / power_before) if power_before > 0 else 0.0
        
        # Phase error RMS (if pristine field available)
        phase_error_rms_before = None
        phase_error_rms_after = None
        reference_field = grid_info.get('pristine_field', None)
        if reference_field is not None:
            # Normalize for phase comparison
            E_rx_norm = received_field / (np.abs(received_field) + 1e-12)
            E_ref_norm = reference_field / (np.abs(reference_field) + 1e-12)
            phase_error_before = np.angle(E_rx_norm * np.conj(E_ref_norm))
            mask = np.abs(received_field) > (np.max(np.abs(received_field)) * 0.1)
            if np.sum(mask) > 0:
                phase_error_rms_before = np.std(phase_error_before[mask])
            
            E_corr_norm = corrected_field / (np.abs(corrected_field) + 1e-12)
            phase_error_after = np.angle(E_corr_norm * np.conj(E_ref_norm))
            mask_after = np.abs(corrected_field) > (np.max(np.abs(corrected_field)) * 0.1)
            if np.sum(mask_after) > 0:
                phase_error_rms_after = np.std(phase_error_after[mask_after])
        
        metrics = {
            'applied': True,
            'method': 'modal',
            # Existing metrics
            'purity_before': sensor.compute_purity_metric(H_before),
            'purity_after': sensor.compute_purity_metric(H_after),
            'cond_before': sensor.compute_condition_metric(H_before),
            'cond_after': sensor.compute_condition_metric(H_after),
            # New comprehensive metrics
            'coupling_efficiency_before': sensor.compute_coupling_efficiency(H_before),
            'coupling_efficiency_after': sensor.compute_coupling_efficiency(H_after),
            'crosstalk_dB_before': sensor.compute_crosstalk_dB(H_before),
            'crosstalk_dB_after': sensor.compute_crosstalk_dB(H_after),
            'crosstalk_reduction_dB': sensor.compute_crosstalk_dB(H_before) - sensor.compute_crosstalk_dB(H_after),
            'power_before': power_before,
            'power_after': power_after,
            'power_loss_dB': power_loss_dB,
            'phase_error_rms_before': phase_error_rms_before,
            'phase_error_rms_after': phase_error_rms_after,
            'phase_error_reduction_rad': (phase_error_rms_before - phase_error_rms_after) if (phase_error_rms_before is not None and phase_error_rms_after is not None) else None,
            # Full crosstalk matrices
            'H_before': H_before,
            'H_after': H_after,
            'zernike_coeffs': controller.coefficients_history[-1] if controller.coefficients_history else {}
        }
    
    elif method == 'sensorless':
        # Sensorless optimization
        optimizer = SensorlessOptimizer(spatial_modes, grid_info, metric='mode_purity')
        corrected_field = optimizer.optimize_correction(received_field)
        
        sensor = ModePuritySensor(spatial_modes, 1550e-9, 25e-3, 1000.0)
        H_before = sensor.measure_crosstalk(received_field, grid_info)
        H_after = sensor.measure_crosstalk(corrected_field, grid_info)
        
        # Power metrics
        delta = grid_info.get('delta', 1.0)
        dA = delta ** 2
        power_before = np.sum(np.abs(received_field) ** 2) * dA
        power_after = np.sum(np.abs(corrected_field) ** 2) * dA
        power_loss_dB = 10 * np.log10(power_after / power_before) if power_before > 0 else 0.0
        
        # Phase error RMS (if pristine field available)
        phase_error_rms_before = None
        phase_error_rms_after = None
        reference_field = grid_info.get('pristine_field', None)
        if reference_field is not None:
            E_rx_norm = received_field / (np.abs(received_field) + 1e-12)
            E_ref_norm = reference_field / (np.abs(reference_field) + 1e-12)
            phase_error_before = np.angle(E_rx_norm * np.conj(E_ref_norm))
            mask = np.abs(received_field) > (np.max(np.abs(received_field)) * 0.1)
            if np.sum(mask) > 0:
                phase_error_rms_before = np.std(phase_error_before[mask])
            
            E_corr_norm = corrected_field / (np.abs(corrected_field) + 1e-12)
            phase_error_after = np.angle(E_corr_norm * np.conj(E_ref_norm))
            mask_after = np.abs(corrected_field) > (np.max(np.abs(corrected_field)) * 0.1)
            if np.sum(mask_after) > 0:
                phase_error_rms_after = np.std(phase_error_after[mask_after])
        
        metrics = {
            'applied': True,
            'method': 'sensorless',
            # Existing metrics
            'purity_before': sensor.compute_purity_metric(H_before),
            'purity_after': sensor.compute_purity_metric(H_after),
            'cond_before': sensor.compute_condition_metric(H_before),
            'cond_after': sensor.compute_condition_metric(H_after),
            # New comprehensive metrics
            'coupling_efficiency_before': sensor.compute_coupling_efficiency(H_before),
            'coupling_efficiency_after': sensor.compute_coupling_efficiency(H_after),
            'crosstalk_dB_before': sensor.compute_crosstalk_dB(H_before),
            'crosstalk_dB_after': sensor.compute_crosstalk_dB(H_after),
            'crosstalk_reduction_dB': sensor.compute_crosstalk_dB(H_before) - sensor.compute_crosstalk_dB(H_after),
            'power_before': power_before,
            'power_after': power_after,
            'power_loss_dB': power_loss_dB,
            'phase_error_rms_before': phase_error_rms_before,
            'phase_error_rms_after': phase_error_rms_after,
            'phase_error_reduction_rad': (phase_error_rms_before - phase_error_rms_after) if (phase_error_rms_before is not None and phase_error_rms_after is not None) else None,
            # Full crosstalk matrices
            'H_before': H_before,
            'H_after': H_after
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corrected_field, metrics


if __name__ == "__main__":
    # Example usage
    print("Adaptive Optics Simulation Module")
    print("=" * 50)
    
    # Test Zernike polynomials
    N = 256
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    # Generate some Zernike modes
    zernike_basis = generate_zernike_basis(6, R, PHI, aperture_radius=1.0)
    print(f"Generated {len(zernike_basis)} Zernike modes")
    
    # Test phase decomposition
    test_phase = 2.0 * zernike_basis[4] + 1.5 * zernike_basis[5]  # Defocus + Astigmatism
    coeffs = decompose_phase_zernike(test_phase, R, PHI, n_modes=6, aperture_radius=1.0, grid_info={'delta': x[1]-x[0]})
    print(f"\nDecomposed phase coefficients:")
    for j, a_j in coeffs.items():
        print(f"  Z{j}: {a_j:.3f}")
    
    print("\n✓ Adaptive Optics module ready for integration!")

