# receiver.py -- Rectified receiver for FSO-MDM OAM system
# Requirements: numpy, scipy, matplotlib, encoding.py, turbulence.py, lgBeam.py (optional)
import os
import sys
import warnings
import argparse
from typing import Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from scipy.fft import fft2, ifft2

# script dir resolution (allow running in notebooks)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

# imports from user modules (encoding provides framing, pilots, LDPC wrapper)
try:
    from encoding import QPSKModulator, PilotHandler, PyLDPCWrapper, FSO_MDM_Frame
except Exception as e:
    raise ImportError(f"receiver.py requires encoding.py in the same directory: {e}")

# optional turbulence angular prop
try:
    from turbulence import angular_spectrum_propagation
except Exception as e:
    angular_spectrum_propagation = None
    warnings.warn(f"turbulence.angular_spectrum_propagation not available: {e}")

# optional lgBeam
try:
    from lgBeam import LaguerreGaussianBeam
except Exception:
    LaguerreGaussianBeam = None

warnings.filterwarnings("ignore")



def reconstruct_grid_from_gridinfo(grid_info: Dict[str, Any]):
    if grid_info is None:
        raise ValueError("grid_info is required to reconstruct spatial grid.")
    x = np.asarray(grid_info.get("x"))
    y = np.asarray(grid_info.get("y"))
    if x.size == 0 or y.size == 0:
        raise ValueError("grid_info.x/y empty or missing.")
    X, Y = np.meshgrid(x, y, indexing="ij")
    delta_x = np.mean(np.diff(x))
    delta_y = np.mean(np.diff(y))
    if not np.isclose(delta_x, delta_y, rtol=1e-6, atol=0.0):
        warnings.warn("Non-square sampling intervals detected; using delta_x as delta.")
    delta = float(delta_x)
    return X, Y, delta, x, y


def energy_normalize_field(field: np.ndarray, delta: float):
    p = np.sum(np.abs(field) ** 2) * (delta ** 2)
    if p > 0:
        return field / np.sqrt(p)
    return field


class OAMDemultiplexer:
    def __init__(self, spatial_modes, wavelength, w0, z_distance, angular_prop_func=angular_spectrum_propagation):
        self.spatial_modes = list(spatial_modes)
        self.n_modes = len(self.spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.angular_prop = angular_prop_func
        self._ref_cache = {}

    def _make_ref_key(self, mode_key, N, delta, X_shape):
        return (mode_key, int(N), float(delta), tuple(X_shape))

    def reference_field(self, mode_key: Tuple[int, int], X, Y, delta, grid_z, tx_beam_obj=None, scaling_factor=None):
        """
        Generate reference field for projection.
        
        Args:
            mode_key: (p, l) mode tuple
            X, Y: Grid coordinates
            delta: Grid spacing
            grid_z: Propagation distance
            tx_beam_obj: Beam object from transmitter (optional)
            scaling_factor: Scaling factor to apply BEFORE propagation (to match TX basis fields)
        
        Returns:
            Reference field at z=grid_z
        """
        N = X.shape[0]
        # Include scaling_factor in cache key to avoid incorrect caching
        cache_key_data = (mode_key, int(N), float(delta), tuple(X.shape), grid_z, scaling_factor)
        key = cache_key_data
        if key in self._ref_cache:
            return self._ref_cache[key].copy()

        R = np.sqrt(X ** 2 + Y ** 2)
        PHI = np.arctan2(Y, X)

        beam = tx_beam_obj
        if beam is None:
            p, l = mode_key
            if LaguerreGaussianBeam is None:
                # fallback analytic simple helix*gaussian (unnormalized)
                w = self.w0
                amp = (R / w) ** (2 * p) * np.exp(-(R ** 2) / (w ** 2))
                ref_z0 = amp * np.exp(1j * l * PHI)
            else:
                beam = LaguerreGaussianBeam(p, l, self.wavelength, self.w0)
                ref_z0 = beam.generate_beam_field(R, PHI, 0.0)
        else:
            ref_z0 = beam.generate_beam_field(R, PHI, 0.0)

        # CRITICAL FIX: Apply scaling BEFORE propagation (to match TX basis fields)
        # TX basis fields are scaled at z=0, then propagated
        # Reference fields should follow the same order
        if scaling_factor is not None:
            ref_z0 = ref_z0 * scaling_factor

        # if grid_z>0 and angular_prop provided, propagate numerically
        if self.angular_prop is None or grid_z == 0.0:
            ref = ref_z0
        else:
            ref = self.angular_prop(ref_z0.copy(), delta, self.wavelength, grid_z)

        self._ref_cache[key] = ref.copy()
        return ref

    def project_field(self, E_rx, grid_info, receiver_radius=None, tx_frame=None):
        X, Y, delta, x, y = reconstruct_grid_from_gridinfo(grid_info)
        R = np.sqrt(X**2 + Y**2)
        
        # CRITICAL: Clear cache if scaling factors are present (to ensure fresh fields with correct scaling)
        # This prevents using cached fields from previous runs with different scaling
        if tx_frame is not None and hasattr(tx_frame, 'metadata') and tx_frame.metadata is not None:
            if 'basis_scaling_factors' in tx_frame.metadata:
                # Clear cache to force regeneration with new scaling factors
                self._ref_cache.clear()

        if not np.iscomplexobj(E_rx):
            warnings.warn("E_rx appears to be real (intensity). Assuming sqrt(I) zero-phase field for projection.")
            E_rx = np.sqrt(np.abs(E_rx)).astype(np.complex128)

        dA = float(delta ** 2)
        if receiver_radius is not None:
            aperture_mask = (R <= receiver_radius).astype(float)
        else:
            aperture_mask = np.ones_like(R, dtype=float)

        symbols = {}
        N = X.shape[0]

        for mode_key in self.spatial_modes:
            tx_beam_obj = None
            scaling_factor = None
            if tx_frame is not None:
                sig = tx_frame.tx_signals.get(mode_key)
                if sig is not None:
                    tx_beam_obj = sig.get("beam", None)
                # Get scaling factor to apply BEFORE propagation (to match TX basis fields)
                if hasattr(tx_frame, 'metadata') and tx_frame.metadata is not None:
                    scaling_factors = tx_frame.metadata.get('basis_scaling_factors', {})
                    if mode_key in scaling_factors:
                        scaling_factor = scaling_factors[mode_key]

           
            ref = self.reference_field(mode_key, X, Y, delta, grid_z=self.z_distance, 
                                      tx_beam_obj=tx_beam_obj, scaling_factor=scaling_factor)
                        
            # CRITICAL FIX: Do NOT apply amplitude_loss to reference field!
            # E_rx already has attenuation applied in pipeline.py
            # Applying it here causes double-attenuation and makes ref_energy too small
            # if tx_frame is not None and hasattr(tx_frame, 'metadata') and tx_frame.metadata is not None:
            #     amplitude_loss = tx_frame.metadata.get('amplitude_loss', 1.0)
            #     ref = ref * amplitude_loss  # ← BUG: Double attenuation!
            

            ref_ap = ref * aperture_mask
            ref_energy = np.sum(np.abs(ref_ap) ** 2) * dA

            E_rx_masked = E_rx * aperture_mask
            projection = np.sum(E_rx_masked * np.conj(ref_ap)) * dA
            
            # DEBUG: Check power matching (only for first mode, first call)
            if mode_key == self.spatial_modes[0] and not hasattr(self, '_debug_printed'):
                # E_rx already has aperture mask applied in pipeline.py, so these should be the same
                E_rx_power = np.sum(np.abs(E_rx_masked) ** 2) * dA
                E_rx_power_no_mask = np.sum(np.abs(E_rx) ** 2) * dA
                ref_power_before_ap = np.sum(np.abs(ref) ** 2) * dA
                ref_power_after_ap = ref_energy
                aperture_efficiency = np.sum(aperture_mask) / (N * N)  # Fraction of pixels in aperture
                print(f"   DEBUG Power Check (mode {mode_key}):")
                print(f"      E_rx power (no mask): {E_rx_power_no_mask:.6e} W")
                print(f"      E_rx power (in aperture): {E_rx_power:.6e} W")
                print(f"      E_ref power (before aperture): {ref_power_before_ap:.6e} W")
                print(f"      E_ref power (after aperture): {ref_power_after_ap:.6e} W")
                print(f"      Aperture efficiency (pixel fraction): {aperture_efficiency:.4f}")
                print(f"      Receiver radius: {receiver_radius:.4f} m" if receiver_radius is not None else "      Receiver radius: None")
                print(f"      Scaling factor: {scaling_factor:.6f}" if scaling_factor is not None else "      Scaling factor: None")
                print(f"      Amplitude loss: {amplitude_loss:.6f}" if 'amplitude_loss' in locals() else "      Amplitude loss: N/A")
                self._debug_printed = True
            
            if ref_energy > 1e-20:
                symbols[mode_key] = projection / ref_energy
            else:
                symbols[mode_key] = 0.0 + 0.0j
        return symbols

    def extract_symbols_sequence(self, E_rx_sequence, grid_info, receiver_radius=None, tx_frame=None):
        seq = np.asarray(E_rx_sequence)
        if seq.ndim == 2:
            seq = seq[np.newaxis, ...]
        n_frames = seq.shape[0]
        symbols_per_mode = {mode: np.zeros(n_frames, dtype=complex) for mode in self.spatial_modes}
        for i in range(n_frames):
            snapshot = self.project_field(seq[i], grid_info, receiver_radius, tx_frame=tx_frame)
            for mode in self.spatial_modes:
                symbols_per_mode[mode][i] = snapshot.get(mode, 0.0 + 0.0j)
        return symbols_per_mode


class ChannelEstimator:
    def __init__(self, pilot_handler: PilotHandler, spatial_modes):
        self.pilot_handler = pilot_handler
        self.spatial_modes = list(spatial_modes)
        self.M = len(self.spatial_modes)
        self.H_est = None
        self.noise_var_est = None

    def _gather_pilots(self, rx_symbols_per_mode: Dict[Tuple[int,int], np.ndarray], tx_frame: FSO_MDM_Frame):
        tx_signals = tx_frame.tx_signals if tx_frame is not None else {}

        pilot_positions = None
        for mode_key in self.spatial_modes:
            sig = tx_signals.get(mode_key)
            if sig is not None and "pilot_positions" in sig:
                pilot_positions = np.asarray(sig["pilot_positions"], dtype=int)
                break
        if pilot_positions is None:
            pilot_positions = np.asarray(self.pilot_handler.pilot_positions, dtype=int) if (self.pilot_handler and getattr(self.pilot_handler, 'pilot_positions', None) is not None) else np.array([], dtype=int)

        if pilot_positions is None or len(pilot_positions) == 0:
            return None, None, np.array([], dtype=int)

        min_len = min([len(rx_symbols_per_mode[mk]) for mk in self.spatial_modes])
        valid_pos = pilot_positions[pilot_positions < min_len]
        if len(valid_pos) == 0:
            return None, None, np.array([], dtype=int)

        n_p = len(valid_pos)
        Y_p = np.zeros((self.M, n_p), dtype=complex)
        P_p = np.zeros((self.M, n_p), dtype=complex)

        for idx, mk in enumerate(self.spatial_modes):
            Y_p[idx, :] = rx_symbols_per_mode[mk][valid_pos]
            if tx_frame is None or mk not in tx_frame.tx_signals:
                raise ValueError("tx_frame with tx_signals required for LS channel estimation (to provide pilot symbols).")
            
            # Use pilot_sequence if available (explicit known pilot values), otherwise fall back to symbols[pilot_pos]
            # After encoding.py fixes, pilots have zero phase noise, so symbols[pilot_pos] should match pilot_sequence
            sig = tx_frame.tx_signals[mk]
            if "pilot_sequence" in sig and sig["pilot_sequence"] is not None:
                pilot_seq = np.asarray(sig["pilot_sequence"], dtype=complex)
                # Map valid_pos to indices in pilot_sequence (valid_pos are positions in full frame)
                # Find which pilots in pilot_sequence correspond to valid_pos
                all_pilot_pos = np.asarray(sig.get("pilot_positions", []), dtype=int)
                if len(all_pilot_pos) > 0 and len(pilot_seq) > 0:
                    # Find indices in all_pilot_pos that match valid_pos
                    pos_to_seq_idx = {pos: i for i, pos in enumerate(all_pilot_pos)}
                    seq_indices = [pos_to_seq_idx[pos] for pos in valid_pos if pos in pos_to_seq_idx]
                    if len(seq_indices) == len(valid_pos):
                        P_p[idx, :] = pilot_seq[seq_indices]
                    else:
                        # Fallback: use symbols[pilot_pos]
                        tx_syms = np.asarray(sig["symbols"])
                        P_p[idx, :] = tx_syms[valid_pos]
                else:
                    tx_syms = np.asarray(sig["symbols"])
                    P_p[idx, :] = tx_syms[valid_pos]
            else:
                # Fallback: extract from symbols array
                tx_syms = np.asarray(sig["symbols"])
                P_p[idx, :] = tx_syms[valid_pos]

        return Y_p, P_p, valid_pos

    def estimate_channel_ls(self, rx_symbols_per_mode: Dict[Tuple[int, int], np.ndarray], tx_frame: FSO_MDM_Frame):
        Y_p, P_p, pilot_pos = self._gather_pilots(rx_symbols_per_mode, tx_frame)
        if Y_p is None or P_p is None or P_p.size == 0:
            warnings.warn("No valid pilots found for LS channel estimation. Returning identity H.")
            self.H_est = np.eye(self.M, dtype=complex)
            return self.H_est

        try:
            PPH = P_p @ P_p.conj().T
            cond = np.linalg.cond(PPH)
            if cond > 1e6:
                warnings.warn(f"Pilot Gram matrix ill-conditioned (cond={cond:.2e}), using pseudo-inverse.")
                H = Y_p @ pinv(P_p)
            else:
                H = Y_p @ P_p.conj().T @ inv(PPH)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix inversion failed; using pseudo-inverse for channel estimate.")
            H = Y_p @ pinv(P_p)

        # NOTE: Removed incorrect regularization (was never triggered, cond(H) < 1e8 in all tests)
        # if np.linalg.cond(H) > 1e8:
        #     reg = 1e-6
        #     H = H @ inv(H + reg * I)  # ← This formula was mathematically wrong

        self.H_est = H
        return H

    def estimate_noise_variance(self, rx_symbols_per_mode: Dict[Tuple[int,int], np.ndarray], tx_frame: FSO_MDM_Frame, H_est: np.ndarray):
        """
        Estimate noise variance for MMSE equalization.
        
        CRITICAL FIX: Use true noise variance from metadata instead of biased residual estimate.
        The residual-based method (Y_p - H_est @ P_p) includes:
        - Channel estimation error
        - Turbulence effects not captured by LS estimate  
        - Actual noise
        This causes massive over-estimation (e.g., 0.43 instead of 1e-9), making MMSE useless.
        
        Solution: Use the true noise variance calculated from SNR in pipeline.py.
        """
        # Check if noise is disabled in metadata
        if hasattr(tx_frame, 'metadata') and tx_frame.metadata is not None:
            noise_disabled = tx_frame.metadata.get('noise_disabled', False)
            print(f"   [DEBUG] noise_disabled flag found: {noise_disabled}")  # DEBUG
            if noise_disabled:
                # Use tiny epsilon for ZF-like behavior (prevents over-regularization)
                self.noise_var_est = 1e-6
                print(f"   [DEBUG] Noise disabled → forcing noise_var = 1e-6 (ZF-like)")  # DEBUG
                return self.noise_var_est
            
            # CRITICAL FIX: Use true noise variance from metadata
            if 'noise_var_per_pixel' in tx_frame.metadata:
                noise_var_per_pixel = tx_frame.metadata['noise_var_per_pixel']
                # Convert per-pixel variance to per-symbol variance
                # The projection sums over pixels, so variance scales with number of pixels
                # But we want the noise variance in the symbol domain after projection
                # For now, use the per-pixel value as a conservative estimate
                self.noise_var_est = max(noise_var_per_pixel, 1e-12)
                snr_db = tx_frame.metadata.get('snr_db', 'unknown')
                print(f"   [DEBUG] Using true noise_var from metadata: {self.noise_var_est:.3e} (SNR={snr_db}dB)")
                return self.noise_var_est
        else:
            print(f"   [DEBUG] No metadata or noise_disabled flag not found")  # DEBUG
        
        # Fallback: estimate from pilot residuals (biased, but better than nothing)
        print(f"   [DEBUG] WARNING: No noise_var in metadata, falling back to residual estimate (biased!)")
        Y_p, P_p, pilot_pos = self._gather_pilots(rx_symbols_per_mode, tx_frame)
        if Y_p is None or P_p is None or P_p.size == 0:
            self.noise_var_est = 1e-6
            return self.noise_var_est
        residual = Y_p - H_est @ P_p
        noise_var = np.mean(np.abs(residual) ** 2)
        self.noise_var_est = max(noise_var, 1e-12)
        print(f"   [DEBUG] Estimated noise_var from residuals (BIASED): {noise_var:.3e}")  # DEBUG
        return self.noise_var_est



class FSORx:
    def __init__(self, spatial_modes, wavelength, w0, z_distance, pilot_handler: PilotHandler, ldpc_instance: Optional[PyLDPCWrapper] = None, eq_method: str = "mmse", receiver_radius: Optional[float] = None):
        self.spatial_modes = list(spatial_modes)
        self.n_modes = len(self.spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.pilot_handler = pilot_handler
        self.eq_method = eq_method.lower()
        self.receiver_radius = receiver_radius

        self.qpsk = QPSKModulator(symbol_energy=1.0)

        if ldpc_instance is not None:
            self.ldpc = ldpc_instance
        else:
            try:
                self.ldpc = PyLDPCWrapper(n=2048, rate=0.8, dv=2, dc=8, seed=42)
                warnings.warn("No LDPC instance provided; receiver created local PyLDPCWrapper that may not match TX.")
            except Exception as e:
                self.ldpc = None
                warnings.warn(f"Cannot construct LDPC wrapper locally: {e}; LDPC decode disabled for demo.")

        self.demux = OAMDemultiplexer(self.spatial_modes, self.wavelength, self.w0, self.z_distance)
        self.chan_est = ChannelEstimator(self.pilot_handler, self.spatial_modes)
        self.metrics = {}

    def receive_frame(self, rx_field_sequence, tx_frame: FSO_MDM_Frame, original_data_bits: np.ndarray, verbose: bool = True, bypass_ldpc: bool = False):
        """
        Main receiver processing pipeline.
        
        Args:
            rx_field_sequence: List/array of 2D complex fields E_rx(x,y) [n_frames, N, N]
            tx_frame: FSO_MDM_Frame from transmitter (must have grid_info and tx_signals)
            original_data_bits: Original INFO bits (before LDPC encoding) for BER calculation.
                               This should be the same bits passed to transmitter.transmit().
            verbose: Print processing steps
            bypass_ldpc: If True, skip LDPC decoding (return coded bits instead of info bits)
        
        Returns:
            decoded_info_bits: Recovered info bits (after LDPC decoding, or coded bits if bypass_ldpc=True)
            metrics: Dictionary with BER, H_est, noise_var, cond_H, etc.
        """
        if verbose:
            print("")

        grid_info = tx_frame.grid_info
        if grid_info is None:
            raise ValueError("tx_frame.grid_info required for demux/projection.")

        if verbose:
            print("1) OAM demultiplexing (projection)...")
        rx_symbols_per_mode = self.demux.extract_symbols_sequence(rx_field_sequence, grid_info, receiver_radius=self.receiver_radius, tx_frame=tx_frame)
        if verbose:
            first_mode = self.spatial_modes[0]
            print(f"   Extracted {len(rx_symbols_per_mode[first_mode])} symbols per mode (incl. pilots).")

        if verbose:
            print("2) Channel estimation (LS using pilots)...")
        H_est = self.chan_est.estimate_channel_ls(rx_symbols_per_mode, tx_frame)
        if verbose:
            print("   H_est magnitude (rows):")
            for row in np.abs(H_est):
                print("     [" + " ".join(f"{v:.3f}" for v in row) + "]")
            print(f"   cond(H_est) = {np.linalg.cond(H_est):.2e}")

        if verbose:
            print("3) Noise variance estimation...")
        noise_var = self.chan_est.estimate_noise_variance(rx_symbols_per_mode, tx_frame, H_est)
        if verbose:
            print(f"   Estimated noise variance σ² = {noise_var:.3e}")

        if verbose:
            print("4) Separate pilots and data")
        pilot_positions = None
        for mk in self.spatial_modes:
            sig = tx_frame.tx_signals.get(mk)
            if sig is not None and "pilot_positions" in sig:
                pilot_positions = np.asarray(sig["pilot_positions"], dtype=int)
                break
        if pilot_positions is None:
            pilot_positions = np.asarray(self.pilot_handler.pilot_positions, dtype=int) if (self.pilot_handler and getattr(self.pilot_handler, 'pilot_positions', None) is not None) else np.array([], dtype=int)

        first_mode = self.spatial_modes[0]
        total_rx_symbols = len(rx_symbols_per_mode[first_mode])
        data_mask = np.ones(total_rx_symbols, dtype=bool)
        if pilot_positions is not None and pilot_positions.size > 0:
            valid_pilots = pilot_positions[pilot_positions < total_rx_symbols]
            data_mask[valid_pilots] = False

        rx_data_per_mode = {mk: rx_symbols_per_mode[mk][data_mask] for mk in self.spatial_modes}
        data_lengths = [len(v) for v in rx_data_per_mode.values()]
        if len(set(data_lengths)) > 1:
            warnings.warn("Uneven data counts across modes; truncating to minimum length.")
            min_len = min(data_lengths)
            for mk in self.spatial_modes:
                rx_data_per_mode[mk] = rx_data_per_mode[mk][:min_len]
        if data_lengths and data_lengths[0] == 0:
            raise ValueError("No data symbols available after removing pilots.")

        Y_data = np.vstack([rx_data_per_mode[mk] for mk in self.spatial_modes])
        N_data = Y_data.shape[1]
        
        if verbose:
            print(f"   Data symbols per mode: {N_data}")

        if verbose:
            print("5) Equalization")
        H = H_est.copy()
        try:
            cond_H = np.linalg.cond(H)
        except Exception:
            cond_H = np.inf
        if self.eq_method == "auto":
            # Use MMSE if condition number is high OR if H values are very small (scaling issues)
            h_max = np.max(np.abs(H))
            h_min = np.min(np.abs(H[np.abs(H) > 1e-10]))  # Min non-zero diagonal
            # Use MMSE if: high condition number, small max H, or very small min H
            use_mmse = (cond_H > 1e4) or (h_max < 0.1) or (h_min < 0.01)
            if verbose and use_mmse:
                print(f"   Auto-selected MMSE (cond={cond_H:.2e}, h_max={h_max:.3f}, h_min={h_min:.3f})")
        else:
            use_mmse = (self.eq_method == "mmse")

        if not use_mmse:
            # ZF (Zero-Forcing) equalization with small regularization for numerical stability
            try:
                # Add small regularization to prevent amplification of small H values
                reg = 1e-6 * np.eye(self.n_modes)
                W_zf = inv(H + reg)
                S_est = W_zf @ Y_data
            except np.linalg.LinAlgError:
                warnings.warn("ZF inversion failed; switching to pseudo-inverse.")
                W_zf = pinv(H)
                S_est = W_zf @ Y_data
        else:
            # MMSE (Minimum Mean Square Error) equalization
            # Standard MMSE formula: W = H^H (H H^H + σ²I)^(-1)
            # This minimizes E[||S - W*Y||²] and is optimal in MSE sense
            sigma2 = max(noise_var, 1e-12)  # Noise variance (ensure non-zero)
            signal_var = 1.0  # Signal variance (QPSK with unit energy)
            reg = sigma2 / signal_var  # Standard MMSE regularization parameter
            
            try:
                # Numerically stable form: W = H^H (H H^H + reg*I)^(-1)
                # This is more stable than W = (H^H H + reg*I)^(-1) H^H when H is ill-conditioned
                W_mmse = H.conj().T @ inv(H @ H.conj().T + reg * np.eye(self.n_modes))
                S_est = W_mmse @ Y_data
                if verbose:
                    print(f"   MMSE regularization: σ²={sigma2:.3e}, reg={reg:.3e}")
            except np.linalg.LinAlgError:
                warnings.warn("MMSE matrix inversion failed; fallback to pseudo-inverse.")
                W_mmse = pinv(H)
                S_est = W_mmse @ Y_data
        
        # CRITICAL FIX: Normalize equalizer output to match QPSK constellation
        # The equalizer may scale symbols arbitrarily due to reference field normalization mismatch
        # QPSK symbols should have average power = 1.0 (magnitude ~0.707)
        # We auto-scale to ensure correct amplitude for demodulation and LLR calculation
        avg_power = np.mean(np.abs(S_est) ** 2)
        if avg_power > 1e-12:  # Avoid division by zero
            scaling_factor = 1.0 / np.sqrt(avg_power)
            S_est_normalized = S_est * scaling_factor
            if verbose:
                print(f"   [DEBUG] Auto-scaling equalizer output:")
                print(f"   [DEBUG]   Raw avg power: {avg_power:.4f} → Scaling by {scaling_factor:.4f}")
                print(f"   [DEBUG]   Normalized avg power: {np.mean(np.abs(S_est_normalized)**2):.4f}")
            S_est = S_est_normalized
        
        # CRITICAL FIX 4: Residual Phase Correction (Blind Carrier Phase Recovery)
        # Atmospheric turbulence adds a random "piston phase" (constant phase offset) to the entire beam
        # This rotates the QPSK constellation, pushing symbols near decision boundaries
        # We use the 4th power method: QPSK^4 removes modulation, leaving only 4*phase_error
        # Then we de-rotate to align constellation with axes
        
        # Flatten for phase estimation (work on all symbols together)
        s_flat = S_est.flatten()
        
        # Estimate phase error using 4th power method
        # For QPSK at angles {45°, 135°, 225°, 315°}, raising to 4th power gives:
        # (e^(j*45°))^4 = e^(j*180°) = -1 (negative real axis)
        # (e^(j*135°))^4 = e^(j*540°) = e^(j*180°) = -1
        # etc. - all QPSK symbols^4 point to 180° (π radians)
        # Any rotation δ becomes 4δ in the 4th power, so we divide by 4 to recover δ
        
        # Calculate 4th power average
        s4_avg = np.mean(s_flat ** 4)
        
        # The target for QPSK^4 is -1 (angle = π = 180°)
        # We want to rotate s4_avg to align with -1
        # The phase error is: angle(s4_avg / (-1)) = angle(s4_avg * (-1)) = angle(-s4_avg)
        # This automatically handles angle wrapping via np.angle()
        phase_est = np.angle(-s4_avg) / 4.0
        
        # De-rotate the constellation
        S_est_corrected = S_est * np.exp(-1j * phase_est)
        
        if verbose:
            print(f"   [DEBUG] Blind phase correction:")
            print(f"   [DEBUG]   QPSK^4 angle: {np.degrees(np.angle(s4_avg)):.2f}° (target: 180°)")
            print(f"   [DEBUG]   Residual phase error: {np.degrees(phase_est):.2f}°")
            print(f"   [DEBUG]   Applying de-rotation...")
        
        S_est = S_est_corrected

        if verbose:
            print(f"   Equalized symbols shape: {S_est.shape} (modes x symbols)")
        if verbose:
            print("6) Demodulation (QPSK)")
        # Flatten in row-major (mode-major) order to match transmitter's symbol distribution
        # S_est is (M × N_data), flattening row-major gives: mode0_symbols, mode1_symbols, ...
        s_est_flat = S_est.flatten(order='C')  # 'C' = row-major = mode-major

        tx_coded_bits = None
        tx_symbol_matrix = None
        if tx_frame is not None:
            try:
                tx_data_matrix = []
                for mk in self.spatial_modes:
                    sig = tx_frame.tx_signals.get(mk)
                    frame_syms = np.asarray(sig.get("frame", sig.get("symbols")))
                    tx_data_matrix.append(frame_syms[data_mask])
                tx_data_matrix = np.vstack(tx_data_matrix)
                tx_symbol_matrix = tx_data_matrix.copy()
                tx_data_flat = tx_data_matrix.flatten(order='C')
                tx_coded_bits = self.qpsk.demodulate_hard(tx_data_flat)
            except Exception:
                tx_coded_bits = None
                tx_symbol_matrix = None

        IDEAL_THRESHOLD = 1e-4
        if noise_var < IDEAL_THRESHOLD:
            if verbose:
                print("   Low noise: hard decisions.")
            received_bits = self.qpsk.demodulate_hard(s_est_flat)
            llrs = None
        else:
            if verbose:
                print("   Using soft LLRs for demodulation.")
            llrs = self.qpsk.demodulate_soft(s_est_flat, noise_var)
            received_bits = (llrs < 0).astype(int)

        if tx_coded_bits is not None:
            compare_len_coded = min(len(tx_coded_bits), len(received_bits))
            if compare_len_coded > 0:
                coded_errors = np.sum(tx_coded_bits[:compare_len_coded] != received_bits[:compare_len_coded])
                coded_ber = coded_errors / compare_len_coded
                self.metrics["coded_ber"] = coded_ber
                if verbose:
                    print(f"   Pre-LDPC coded BER: {coded_ber:.3e}")
                    tx_preview = tx_data_flat[:6] if 'tx_data_flat' in locals() else None
                    rx_preview = s_est_flat[:6]
                    if tx_preview is not None:
                        print(f"   Sample TX symbols (mode-major): {tx_preview}")
                    print(f"   Sample RX symbols (mode-major): {rx_preview}")
                    print(f"   TX coded bits len: {len(tx_coded_bits)}, RX coded bits len: {len(received_bits)}")
                    print(f"   TX bits preview: {tx_coded_bits[:12]}")
                    print(f"   RX bits preview: {received_bits[:12]}")

        if verbose:
            print(f"   Demodulated coded bits: {len(received_bits)}")

        if verbose:
            print("7) LDPC decoding")
        decoded_info_bits = np.array([], dtype=int)
        if bypass_ldpc or (self.ldpc is None):
            # bypass or ldpc not available: return coded bits directly (useful for demo)
            decoded_info_bits = received_bits.copy()
            if verbose:
                print(f"   LDPC bypass: returning {len(decoded_info_bits)} bits (no decoding)")
        else:
            try:
                if llrs is not None:
                    decoded_info_bits = self.ldpc.decode_bp(llrs)
                    if verbose:
                        print(f"   Decoded info bits (BP): {len(decoded_info_bits)}")
                    effective_len = int(len(llrs) * (self.ldpc.k / self.ldpc.n))
                else:
                    decoded_info_bits = self.ldpc.decode_hard(received_bits)
                    if verbose:
                        print(f"   Decoded info bits (hard): {len(decoded_info_bits)}")
                    effective_len = int(len(received_bits) * (self.ldpc.k / self.ldpc.n))

                if len(decoded_info_bits) != effective_len:
                    raise ValueError(
                        f"LDPC decoder returned {len(decoded_info_bits)} bits, expected {effective_len}. "
                        "Check pilot/data separation and LDPC block configuration."
                    )

                orig_len = len(original_data_bits) if original_data_bits is not None else len(decoded_info_bits)
                if len(decoded_info_bits) >= orig_len:
                    decoded_info_bits = decoded_info_bits[:orig_len]
                else:
                    raise ValueError(
                        f"Decoded bitstream shorter than original ({len(decoded_info_bits)} < {orig_len})."
                    )
            except Exception as e:
                warnings.warn(f"LDPC decode failed: {e}; falling back to hard bits.")
                decoded_info_bits = received_bits

        if verbose:
            print("8) Performance metrics (BER)")
        orig = np.asarray(original_data_bits, dtype=int)
        L_orig = len(orig)
        L_rec = len(decoded_info_bits)
        compare_len = min(L_orig, L_rec)
        if compare_len == 0 and L_orig > 0:
            bit_errors = L_orig
            ber = 1.0
        else:
            trimmed_orig = orig[:compare_len]
            trimmed_rec = decoded_info_bits[:compare_len]
            bit_errors_common = np.sum(trimmed_orig != trimmed_rec) if compare_len > 0 else 0
            len_mismatch = abs(L_orig - L_rec)
            bit_errors = int(bit_errors_common + len_mismatch)
            ber = bit_errors / L_orig if L_orig > 0 else 0.0

        coded_ber_metric = self.metrics.get("coded_ber") if isinstance(self.metrics, dict) else None
        self.metrics = {
            "H_est": H_est,
            "noise_var": noise_var,
            "bit_errors": int(bit_errors),
            "total_bits": int(L_orig),
            "ber": float(ber),
            "n_data_symbols": int(N_data),
            "n_modes": int(self.n_modes),
            "cond_H": float(np.linalg.cond(H_est))
        }
        if coded_ber_metric is not None:
            self.metrics["coded_ber"] = float(coded_ber_metric)

        # Store symbol samples for diagnostic plots
        sample_lim = min(N_data, 512)
        if sample_lim > 0:
            self.metrics["rx_symbols_sample"] = S_est[:, :sample_lim].copy()
            if tx_symbol_matrix is not None:
                self.metrics["tx_symbols_sample"] = tx_symbol_matrix[:, :sample_lim].copy()
            self.metrics["symbol_sample_count"] = int(sample_lim)

        if verbose:
            print(f"   Original bits: {L_orig}, Decoded bits: {L_rec}, Errors: {bit_errors}, BER={ber:.3e}")

        return decoded_info_bits, self.metrics

    def receive_sequence(self, E_rx_sequence, grid_info=None, tx_signals=None, original_data_bits=None, tx_frame=None, verbose=True):
        """
        Wrapper method for compatibility with pipeline.py.
        
        This method provides an interface that matches pipeline.py's expected call signature.
        It constructs an FSO_MDM_Frame from the provided arguments and calls receive_frame().
        
        Args:
            E_rx_sequence: List/array of 2D complex fields E_rx(x,y)
            grid_info: Spatial grid parameters (dict with 'x', 'y', etc.)
            tx_signals: Dictionary of tx signals per mode (if tx_frame not provided)
            original_data_bits: Original INFO bits (before LDPC encoding) for BER calculation
            tx_frame: FSO_MDM_Frame object (if provided, grid_info and tx_signals are ignored)
            verbose: Print processing steps
        
        Returns:
            decoded_info_bits: Recovered info bits
            metrics: Dictionary with BER, H_est, noise_var, etc.
        """
        # If tx_frame is provided, use it directly
        if tx_frame is not None:
            if original_data_bits is None:
                raise ValueError("original_data_bits required for BER calculation")
            return self.receive_frame(E_rx_sequence, tx_frame, original_data_bits, verbose=verbose, bypass_ldpc=False)
        
        # Otherwise, construct FSO_MDM_Frame from grid_info and tx_signals
        if grid_info is None:
            raise ValueError("Either tx_frame or grid_info must be provided")
        if tx_signals is None:
            raise ValueError("Either tx_frame or tx_signals must be provided")
        if original_data_bits is None:
            raise ValueError("original_data_bits required for BER calculation")
        
        # Construct FSO_MDM_Frame
        from encoding import FSO_MDM_Frame
        tx_frame = FSO_MDM_Frame(tx_signals=tx_signals, grid_info=grid_info)
        
        return self.receive_frame(E_rx_sequence, tx_frame, original_data_bits, verbose=verbose, bypass_ldpc=False)



def plot_constellation(rx_symbols, title="Received Constellation"):
    plt.figure(figsize=(5, 5))
    plt.plot(np.real(rx_symbols), np.imag(rx_symbols), ".", alpha=0.6)
    plt.axhline(0, color="grey")
    plt.axvline(0, color="grey")
    plt.title(title)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receiver demo for FSO-MDM OAM (lightweight).")
    parser.add_argument("--realistic", action="store_true", help="Do NOT orthonormalize spatial refs (use realistic, possibly ill-conditioned refs).")
    parser.add_argument("--debug", action="store_true", help="Enable a few debug prints.")
    parser.add_argument("--use-ldpc", action="store_true", help="Try to run LDPC decoding (requires transmitter LDPC match).")
    parser.add_argument("--snr", type=float, default=25.0, help="SNR in dB for synthesized AWGN.")
    parser.add_argument("--nframes", type=int, default=256, help="Number of spatial frames to synthesize.")
    args, unknown = parser.parse_known_args()

    # lightweight synth function used by demo (vectorized)
    def synthesize_spatial_sequence(tx_frame, spatial_modes, demuxer, n_samples=256, snr_db=30, apply_turb=False, angular_prop=None, orthonormalize_refs=True):
        grid_info = tx_frame.grid_info
        X, Y, delta, x, y = reconstruct_grid_from_gridinfo(grid_info)
        N = X.shape[0]
        M = len(spatial_modes)

        lengths = [len(tx_frame.tx_signals[mk]['symbols']) for mk in spatial_modes]
        total_len = min(lengths)
        T = min(total_len, n_samples)

        # fetch reference fields
        refs_list = []
        for mk in spatial_modes:
            sig = tx_frame.tx_signals.get(mk, {})
            beam_obj = sig.get("beam", None)
            rf = demuxer.reference_field(mk, X, Y, delta, grid_z=demuxer.z_distance, tx_beam_obj=beam_obj)
            refs_list.append(rf)
        refs_stack = np.stack(refs_list, axis=0)  # (M,N,N)

        # Optionally orthonormalize w.r.t area-weighted inner product
        if orthonormalize_refs:
            V = refs_stack.reshape(M, -1).T  # (N*N, M)
            Q, Rq = np.linalg.qr(V, mode='reduced')
            ortho_stack = Q.T.reshape(M, N, N).copy()
            # area normalize
            for m in range(M):
                col = ortho_stack[m]
                norm = np.sqrt(np.sum(np.abs(col)**2) * (delta**2))
                if norm == 0:
                    raise RuntimeError("Zero-energy reference after orthonormalization")
                ortho_stack[m] = col / norm
            refs_stack = ortho_stack
            # refill demux cache with orthonormal refs
            for mi, mk in enumerate(spatial_modes):
                key = demuxer._make_ref_key(mk, N, delta, X.shape)
                demuxer._ref_cache[key] = refs_stack[mi].copy()

        # prepare tx symbol matrix
        tx_sym_matrix = np.zeros((M, T), dtype=np.complex128)
        for mi, mk in enumerate(spatial_modes):
            tx_sym_matrix[mi, :] = np.asarray(tx_frame.tx_signals[mk]['symbols'])[:T]

        rx_fields = np.zeros((T, N, N), dtype=np.complex128)
        snr_lin = 10.0 ** (snr_db / 10.0)

        for t in range(T):
            s_vec = tx_sym_matrix[:, t]
            E = np.tensordot(s_vec, refs_stack, axes=(0, 0))
            if apply_turb and (angular_prop is not None):
                try:
                    E = angular_prop(E, delta, demuxer.wavelength, demuxer.z_distance)
                except Exception as e:
                    print("  Warning: angular_prop failed; continuing: ", e)

            total_power = np.sum(np.abs(E)**2) * (delta**2)
            if total_power <= 0:
                total_power = 1.0
            E = E / np.sqrt(total_power)

            noise_power_total = 1.0 / max(snr_lin, 1e-12)
            noise_var_per_sample = noise_power_total / (N * N)
            sigma = np.sqrt(noise_var_per_sample / 2.0)
            noise = sigma * (np.random.randn(N, N) + 1j * np.random.randn(N, N))

            E_noisy = E + noise
            rx_fields[t] = E_noisy

        return rx_fields, tx_sym_matrix, refs_stack


    from encoding import FSO_MDM_Frame

    N_demo = 256
    D_demo = 0.45
    x = np.linspace(-D_demo/2, D_demo/2, N_demo)
    y = x.copy()
    grid_info = {"x": x, "y": y, "N": N_demo, "D": D_demo}

    spatial_modes = [(0,1),(0,-1),(0,2),(0,-2)]
    frame_len = 512
    pilot_count = 64
    pilot_positions = np.arange(pilot_count)
    rng = np.random.RandomState(42)
    const = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex)/np.sqrt(2)

    tx_signals = {}
    M = len(spatial_modes)
    # create orthogonal pilots (DFT rows across pilot_count) to give well-conditioned P_p
    pilots_matrix = np.exp(1j * 2.0 * np.pi * (np.arange(M)[:, None] * np.arange(pilot_count)[None, :]) / float(pilot_count))

    for mi, mk in enumerate(spatial_modes):
        syms = const[rng.randint(0,4, frame_len)].astype(complex)
        syms[pilot_positions] = pilots_matrix[mi, :]
        tx_signals[mk] = {"symbols": syms, "pilot_positions": list(pilot_positions), "n_symbols": int(len(syms)), "beam": None}

    tx_frame = FSO_MDM_Frame(tx_signals, multiplexed_field=None, grid_info=grid_info, metadata={})

    spatial_modes = list(tx_frame.metadata["spatial_modes"])
    wavelength = getattr(tx_frame, "wavelength", 1550e-9)
    w0 = getattr(tx_frame, "w0", 25e-3)
    z_distance = getattr(tx_frame, "z_distance", 0.0)

    demux = OAMDemultiplexer(spatial_modes, wavelength, w0, z_distance, angular_prop_func=angular_spectrum_propagation)
    class _StubPilot:
        def __init__(self, pos):
            self.pilot_positions = np.asarray(pos, dtype=int)
    pilot_handler = _StubPilot(tx_frame.metadata.get("pilot_positions", list(pilot_positions)))

    ldpc_inst = getattr(tx_frame, "ldpc", None) if args.use_ldpc else None
    fsorx = FSORx(spatial_modes, wavelength, w0, z_distance, pilot_handler, ldpc_instance=ldpc_inst, eq_method="mmse")

    print(f"\nSynthesizing {args.nframes} spatial frames at SNR={args.snr} dB ...")
    rx_fields, tx_sym_matrix, refs_stack = synthesize_spatial_sequence(tx_frame, spatial_modes, demux, n_samples=args.nframes, snr_db=args.snr, apply_turb=False, angular_prop=angular_spectrum_propagation, orthonormalize_refs=not args.realistic)

    # quick Gram check
    M = len(spatial_modes)
    V_ortho = refs_stack.reshape(M, -1)
    G = V_ortho @ V_ortho.conj().T * ((x[1]-x[0])**2)
    print("\nOrthonormalized spatial Gram (real-rounded):")
    print(np.round(G.real, 6))

    # noiseless sanity test
    try:
        s0 = tx_sym_matrix[:, 0]
        E0 = np.tensordot(s0, refs_stack, axes=(0, 0))
        proj0 = demux.project_field(E0, tx_frame.grid_info, tx_frame=tx_frame)
        proj_vec = np.array([proj0[mk] for mk in spatial_modes])
        print("\nNoiseless test: TX s0:", s0)
        print("Projected s0:", proj_vec)
    except Exception as e:
        print("Noiseless sanity check failed:", e)

    # Build original coded bits from TX symbols via QPSK hard demap (option A)
    qpsk = QPSKModulator(symbol_energy=1.0)

    # Determine pilot positions (same as receiver)
    pilot_positions = np.asarray(tx_frame.metadata.get("pilot_positions", list(pilot_positions)), dtype=int)
    T_sent = tx_sym_matrix.shape[1]

    # Build a data mask: True for data, False for pilot (same logic used by FSORx)
    data_mask = np.ones(T_sent, dtype=bool)
    if pilot_positions.size > 0:
        valid_pilots = pilot_positions[pilot_positions < T_sent]
        data_mask[valid_pilots] = False

    # Extract tx data symbols (mode-major order) and flatten
    tx_data_symbols = tx_sym_matrix[:, data_mask]   # shape (M, N_data)
    tx_data_flat = tx_data_symbols.flatten(order='C')  # mode-major flatten

    # Demap to coded bits (these are the "original_data_bits" for receiver BER calc)
    original_coded_bits = qpsk.demodulate_hard(tx_data_flat)

    if args.debug:
        print("\nDEBUG: tx_data_symbols_flat (first 8):", tx_data_flat[:8])
        print("DEBUG: original_coded_bits (len):", len(original_coded_bits))

    # Run the receiver (bypass LDPC by default in demo)
    decoded_bits, metrics = fsorx.receive_frame(rx_fields, tx_frame, original_coded_bits, verbose=not args.debug, bypass_ldpc=True)

    # diagnostics plots (safe-guarded)
    try:
        plt.figure(figsize=(6,5))
        plt.title("received intensity (frame 0)")
        plt.imshow(np.abs(rx_fields[0])**2, origin="lower")
        plt.colorbar(label="Intensity [a.u.]")
        plt.show()
    except Exception:
        pass

    try:
        rx_symbols = demux.extract_symbols_sequence(rx_fields, tx_frame.grid_info, tx_frame=tx_frame)
        first_mode = spatial_modes[0]
        sample_syms = rx_symbols[first_mode][:128]
        plot_constellation(sample_syms, title=f"Projected symbols (mode {first_mode})")
    except Exception:
        pass

    print("\nSanity check complete. Receiver metrics (if available):")
    for k, v in metrics.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape {getattr(v,'shape', None)}")
        else:
            print(f"  {k}: {v}")