import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, eval_genlaguerre
import os

# Handle numpy version compatibility for trapz/trapezoid
_trapezoid = getattr(np, "trapezoid", np.trapz)

class LaguerreGaussianBeam:
    """
    Physically accurate Laguerre-Gaussian (LG) beam model.
    
    Correction Note: 
    This class distinguishes between the 'Fundamental Gaussian Parameter' w(z)
    used for field generation, and the 'Effective Beam Radius' used for 
    geometric loss estimates. This prevents double-counting beam divergence.
    """
    
    def __init__(self, p, l, wavelength, w0):
        if p < 0:
            raise ValueError(f"Radial index p must be non-negative, got p={p}")
        
        self.p = int(p)
        self.l = int(l)
        self.wavelength = float(wavelength)
        self.w0 = float(w0) 
        
        # Wavenumber
        self.k = 2 * np.pi / self.wavelength
        
        # Beam Quality Factor (M^2)
        # This determines how much wider the LG mode is compared to a fundamental Gaussian
        self.M_squared = 2 * self.p + abs(self.l) + 1
        
        # Fundamental Rayleigh Range (Based on w0 ONLY)
        # Literature: Siegman, Lasers, Ch 16-17
        self.z_R = (np.pi * self.w0**2) / self.wavelength
        
        # Normalization Constant
        # Ensures Integral(|E|^2) = 1 over infinite plane
        self.C_norm = np.sqrt(2.0 * factorial(self.p) / (np.pi * factorial(self.p + abs(self.l))))

    def beam_waist(self, z):
        """
        Calculates w(z) for the FUNDAMENTAL Gaussian mode.
        NOTE: This is NOT the physical radius of the LG beam. 
        It is the scaling parameter for the Laguerre polynomials.
        """
        return self.w0 * np.sqrt(1 + (z / self.z_R) ** 2)

    def physical_beam_radius(self, z):
        """
        Calculates the effective physical radius of the beam (approx D4sigma).
        Use THIS value for geometric loss / aperture clipping calculations.
        Formula: w_physical(z) = w_fundamental(z) * sqrt(M^2)
        """
        return self.beam_waist(z) * np.sqrt(self.M_squared)

    def radius_of_curvature(self, z):
        """Radius of curvature of the wavefront at distance z."""
        z = np.asarray(z)
        R_z = np.full_like(z, np.inf, dtype=float)
        nonzero_mask = np.abs(z) >= 1e-12
        
        # Standard Gaussian formula
        R_z[nonzero_mask] = z[nonzero_mask] * (1 + (self.z_R / z[nonzero_mask]) ** 2)
        
        if np.ndim(z) == 0:
            return float(R_z)
        return R_z

    def gouy_phase(self, z):
        """
        Accumulated Gouy phase shift.
        For LG modes, this is (2p + |l| + 1) times the fundamental shift.
        """
        # The M_squared factor applies to the PHASE
        return self.M_squared * np.arctan(z / self.z_R)

    @property
    def effective_divergence_angle(self):
        """
        Returns (Fundamental Divergence, Effective/Physical Divergence)
        """
        theta_0 = self.wavelength / (np.pi * self.w0)
        theta_eff = theta_0 * np.sqrt(self.M_squared)
        return theta_0, theta_eff

    def generate_beam_field(self, r, phi, z, P_tx_watts=1.0, 
                           laser_linewidth_kHz=None, timing_jitter_ps=None,
                           tx_aperture_radius=None, beam_tilt_x_rad=0.0, beam_tilt_y_rad=0.0,
                           phase_noise_samples=None, symbol_time_s=None):
        """
        Generates the complex electric field E(r, phi, z).
        """
        if np.ndim(z) != 0:
            raise ValueError("z must be a scalar. Use loops for multiple planes.")
        z = float(z)

        # Ensure inputs are arrays
        r = np.asarray(r)
        phi = np.asarray(phi)
        try:
            r, phi = np.broadcast_arrays(r, phi)
        except ValueError:
            raise ValueError("r and phi must be broadcastable to the same shape")
        
        # 1. Get Fundamental Parameters at z
        w_z = self.beam_waist(z)          # Fundamental scale
        R_z = self.radius_of_curvature(z) # Curvature
        psi_z = self.gouy_phase(z)        # Gouy Phase
        
        # 2. Laguerre Polynomial Term
        # Argument is 2r^2 / w^2
        arg = 2 * r**2 / w_z**2
        L_p_l = eval_genlaguerre(self.p, abs(self.l), arg)
        
        # 3. Amplitude Scaling
        power_scale = np.sqrt(P_tx_watts) if P_tx_watts > 0 else 0.0
        amplitude_factor = self.C_norm * (1.0 / w_z) * power_scale

        # 4. Radial Term
        # (sqrt(2)r / w)^|l| * L(...) * exp(-r^2/w^2)
        radial_factor = (
            ((np.sqrt(2) * r) / w_z) ** abs(self.l) * L_p_l * np.exp(-(r**2) / w_z**2)
        )
        
        # 5. Azimuthal Term (OAM Phase)
        azimuthal_factor = np.exp(-1j * self.l * phi)
        
        # 6. Wavefront Curvature Term
        if np.isinf(R_z):
            curvature_factor = 1.0 
        else:
            curvature_phase = -1j * self.k * r**2 / (2 * R_z)
            curvature_factor = np.exp(curvature_phase)
            
        # 7. Gouy Phase Term
        gouy_phase_term = np.exp(-1j * psi_z)
        
        # 8. Beam Steering (Tilt)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        steering_phase = self.k * (x * beam_tilt_x_rad + y * beam_tilt_y_rad)
        
        # 9. Phase Noise & Jitter (Simulated)
        phase_noise = 0.0
        # (Handling explicit sample injection)
        if phase_noise_samples is not None:
            phase_noise = float(phase_noise_samples)
        # (Handling on-the-fly generation)
        elif laser_linewidth_kHz is not None and laser_linewidth_kHz > 0:
            if symbol_time_s is None:
                # Default to 1ns if not provided, just to avoid crash, but warn technically
                symbol_time_s = 1e-9 
            delta_nu = laser_linewidth_kHz * 1e3  
            sigma = np.sqrt(2 * np.pi * delta_nu * symbol_time_s)
            phase_noise = np.random.normal(0, sigma)

        timing_jitter_phase = 0.0
        if timing_jitter_ps is not None and timing_jitter_ps > 0:
            f_carrier = 3e8 / self.wavelength
            jitter_phase_error = 2 * np.pi * timing_jitter_ps * 1e-12 * f_carrier
            timing_jitter_phase = np.random.normal(0, jitter_phase_error)
            
        # 10. Combine Everything
        propagation_phase = np.exp(1j * (self.k * z + phase_noise + timing_jitter_phase))

        field = (
            amplitude_factor
            * radial_factor
            * azimuthal_factor
            * curvature_factor
            * np.exp(1j * steering_phase) 
            * gouy_phase_term
            * propagation_phase
        )
        
        # 11. TX Aperture Clipping (if applied at source)
        if tx_aperture_radius is not None:
            mask = (r <= tx_aperture_radius).astype(float)
            field = field * mask

        return field

    def calculate_intensity(self, r, phi, z, P_tx_watts=1.0, **kwargs):
        """Returns |E|^2"""
        field = self.generate_beam_field(r, phi, z, P_tx_watts=P_tx_watts, **kwargs)
        return np.abs(field) ** 2
    
    def radial_intensity(self, r, z, P_tx_watts=1.0):
        """
        Computes azimuthally-averaged radial intensity profile I(r) = <|E(r,φ)|²>_φ.
        
        For LG modes, this is analytically: I(r) = |C_norm|² (1/w_z)² (2r²/w_z²)^|l| 
        [L_p^|l|(2r²/w_z²)]² exp(-2r²/w_z²) * P_tx_watts
        
        This is much more efficient than generating 2D fields and averaging.
        
        Args:
            r: Radial coordinate(s) [m] (scalar or array)
            z: Propagation distance [m] (scalar)
            P_tx_watts: Transmit power [W]
            
        Returns:
            I(r): Radial intensity profile [W/m²] (same shape as r)
        """
        r = np.asarray(r)
        w_z = self.beam_waist(z)
        
        # Normalization factor (includes power scaling)
        power_scale = P_tx_watts if P_tx_watts > 0 else 0.0
        norm_factor = (self.C_norm ** 2) * (1.0 / (w_z ** 2)) * power_scale
        
        # Radial coordinate normalized to beam waist
        rho_sq = 2.0 * (r ** 2) / (w_z ** 2)
        rho_sq = np.maximum(rho_sq, 0.0)  # Ensure non-negative
        
        # Laguerre polynomial term
        L_p_l = eval_genlaguerre(self.p, abs(self.l), rho_sq)
        
        # Radial intensity: (2r²/w²)^|l| * [L_p^|l|]² * exp(-2r²/w²)
        radial_term = (rho_sq ** abs(self.l)) * (L_p_l ** 2) * np.exp(-rho_sq)
        
        intensity = norm_factor * radial_term
        return intensity
    
    def generate_phase_noise_sequence(self, num_symbols, symbol_time_s, laser_linewidth_kHz, seed=None):
        """Generates a random walk phase noise sequence for time-series simulation."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        if laser_linewidth_kHz is None or laser_linewidth_kHz <= 0:
            return np.zeros(num_symbols)
        
        delta_nu = laser_linewidth_kHz * 1e3 
        phase_variance = 2 * np.pi * delta_nu * symbol_time_s
        
        phase_increments = rng.normal(0, np.sqrt(phase_variance), num_symbols)
        phase_noise = np.cumsum(phase_increments)
        return phase_noise
    
    def overlap_with(self, other, r_max_factor=6.0, n_r=800, n_phi=360):
        """Calculates inner product <Self | Other> to check orthogonality."""
        w_max = max(self.beam_waist(0), other.beam_waist(0))
        r_max = w_max * r_max_factor
        
        r = np.linspace(0.0, r_max, n_r)
        phi = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)
        R, PHI = np.meshgrid(r, phi, indexing='xy') 
        
        # P_tx=1.0 ensures normalized calculation
        E1 = self.generate_beam_field(R, PHI, 0.0, P_tx_watts=1.0)
        E2 = other.generate_beam_field(R, PHI, 0.0, P_tx_watts=1.0)
        
        # Integration measure: r dr dphi
        integrand = np.conjugate(E1) * E2 * R
        
        int_over_phi = _trapezoid(integrand, phi, axis=0)
        overlap = _trapezoid(int_over_phi, r)
        
        return complex(overlap)

    def get_beam_parameters(self, z):
        """Dictionary of beam parameters at distance z."""
        return {
            "z": z,
            "w_z_fundamental": self.beam_waist(z),
            "w_z_physical": self.physical_beam_radius(z),
            "R_z": self.radius_of_curvature(z),
            "gouy_phase": self.gouy_phase(z),
            "M_squared": self.M_squared
        }

    def propagation_summary(self, z_distances):
        print(f"\nPropagation Summary for LG_{self.p}^{self.l} beam:")
        print(f"  w0 = {self.w0*1e3:.1f} mm, z_R = {self.z_R:.1f} m, M² = {self.M_squared}")
        print("-" * 80)
        print(f"{'Dist (m)':<10} | {'w_fund (mm)':<12} | {'w_phys (mm)':<12} | {'R(z) (m)':<10} | {'Gouy (rad)':<10}")
        print("-" * 80)

        for z in z_distances:
            params = self.get_beam_parameters(z)
            w_fund = params["w_z_fundamental"] * 1e3
            w_phys = params["w_z_physical"] * 1e3
            R_z = params["R_z"]
            R_str = "∞" if np.isinf(R_z) else f"{R_z:.2f}"

            print(
                f"{z:<10.1f} | {w_fund:<12.2f} | {w_phys:<12.2f} | {R_str:<10} | {params['gouy_phase']:<10.3f}"
            )

    def __str__(self):
        return (
            f"LG_{self.p}^{self.l} beam (λ={self.wavelength * 1e9:.0f}nm, "
            f"w0={self.w0 * 1e3:.1f}mm, z_R={self.z_R:.1f}m, M²={self.M_squared})"
        )

    # ... (get_tx_parameters_summary remains unchanged, it is purely logging)
    def get_tx_parameters_summary(self, P_tx_watts=1.0, laser_linewidth_kHz=None, 
                                  timing_jitter_ps=None, tx_aperture_radius=None,
                                  beam_tilt_x_rad=0.0, beam_tilt_y_rad=0.0):
        # Implementation identical to previous provided version
        # Included for completeness of file
        summary = {
            'P_tx_watts': P_tx_watts,
            'P_tx_dBm': 10 * np.log10(P_tx_watts * 1000) if P_tx_watts > 0 else -np.inf,
            'laser_linewidth_kHz': laser_linewidth_kHz if laser_linewidth_kHz is not None else 0.0,
            'timing_jitter_ps': timing_jitter_ps if timing_jitter_ps is not None else 0.0,
            'tx_aperture_radius_m': tx_aperture_radius if tx_aperture_radius is not None else np.inf,
            'beam_tilt_x_rad': beam_tilt_x_rad,
            'beam_tilt_y_rad': beam_tilt_y_rad,
            'beam_tilt_x_deg': np.degrees(beam_tilt_x_rad),
            'beam_tilt_y_deg': np.degrees(beam_tilt_y_rad),
        }
        return summary


def plot_beam_analysis(beam, grid_size, max_radius_mm, save_fig=False, plot_dir="plots"):
    """
    Visualization tool. 
    Requires correct 'physical_beam_radius' for sizing plots.
    """
    
    
    figures = []
    
    # Grid setup based on physical size
    r_max_m = max_radius_mm * 1e-3
    x = np.linspace(-r_max_m, r_max_m, grid_size)
    y = np.linspace(-r_max_m, r_max_m, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X) 
    
    # 1. Transverse Profile
    zField = 1000
    field_z = beam.generate_beam_field(R, PHI, zField)
    intensity_z = np.abs(field_z) ** 2 
    phase_z = np.angle(field_z) 
    extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]
    
    output_dir = os.path.join(plot_dir, f"lgBeam_p{beam.p}_l{beam.l}")
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)

    # Plot Intensity
    fig_int, ax_int = plt.subplots(figsize=(6, 5))
    im = ax_int.imshow(intensity_z, extent=extent_mm, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax_int, label='Intensity')
    ax_int.set_title(f"Intensity at z={zField}m (w_phys={beam.physical_beam_radius(zField)*1e3:.1f}mm)")
    figures.append(fig_int)
    
    if save_fig:
        fig_int.savefig(os.path.join(output_dir, "intensity.png"), dpi=300)
        print(f"Saved plots to {output_dir}")

    return figures


def main():
    print("=== LG Beam Verification Mode ===")
    
    # Parameters matching your simulation
    WAVELENGTH = 1550e-9  
    W0 = 25e-3
    
    # Test a high order mode (the one that failed before)
    P_MODE = 0         
    L_MODE = -4          

    beam = LaguerreGaussianBeam(P_MODE, L_MODE, WAVELENGTH, W0)
    print(f"\nDefined Beam: {beam}")
    print(f"Rayleigh Range z_R: {beam.z_R:.2f} m (Should be ~1266m)")

    print("\n--- Propagation Check at 1000m ---")
    z_check = 1000.0
    w_fund = beam.beam_waist(z_check)
    w_phys = beam.physical_beam_radius(z_check)
    
    print(f"At z = {z_check} m:")
    print(f"  Fundamental Param w(z): {w_fund*1e3:.2f} mm")
    print(f"  Physical Radius (D4σ):  {w_phys*1e3:.2f} mm")
    
    # Verification Logic
    if z_check < beam.z_R:
        print("  STATUS: Near Field (z < z_R)")
        # In Near field, size should be roughly w0 * sqrt(M^2)
        expected_approx = W0 * np.sqrt(beam.M_squared)
        print(f"  Expected approx size:   {expected_approx*1e3:.2f} mm")
    else:
        print("  STATUS: Far Field")

    if w_phys < 0.25: # 250mm
        print("\nBeam fits inside 250mm aperture.")
    else:
        print("\nBeam is too large (Error persists).")

if __name__ == "__main__":
    main()