import os
import sys

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from contextlib import redirect_stdout
import io
import argparse
warnings.filterwarnings('ignore')

# Import all modules
try:
    from lgBeam import LaguerreGaussianBeam
    from encoding import encodingRunner, QPSKModulator, PilotHandler
    from fsplAtmAttenuation import calculate_kim_attenuation, calculate_geometric_loss

    from turbulence import (AtmosphericTurbulence,
                            create_multi_layer_screens,
                            apply_multi_layer_turbulence)
    from receiver import FSORx
    
    # Optional: Adaptive Optics
    try:
        from adaptiveOptics import apply_adaptive_optics
        _HAS_AO = True
    except ImportError:
        _HAS_AO = False
        warnings.warn("adaptiveOptics.py not found. AO features disabled.")
except ImportError as e:
    print(f"✗ E2E Simulation Import Error: {e}")
    print("  Please ensure lgBeam.py, encoding.py, fsplAtmAttenuation.py, turbulence.py, and receiver.py are in the same directory.")
    sys.exit(1)

np.random.seed(42)


class SimulationConfig:
    WAVELENGTH = 1550e-9  # [m]
    W0 = 25e-3           # [m]
    DISTANCE = 1000      # [m]
    RECEIVER_DIAMETER = 0.5  # [m] - Fixed: was 0.3m, increased to reduce geometric loss for high-order modes
    P_TX_TOTAL_W = 1.0  # [W] - Fixed: was 0.5, now matches runner.py for consistency
    SPATIAL_MODES = [(0, -1), (0, 1), (0, -3), (0, 3), (0, -4), (0, 4), (1, -1), (1, 1)]
    CN2 = 1e-15  # [m^(-2/3)] - Weak turbulence (realistic)
    L0 = 10.0           # [m]
    L0_INNER = 0.005    # [m]
    NUM_SCREENS = 25
    CN2_MODEL = "uniform"  # Horizontal path → uniform profile; set "hufnagel_valley" for vertical links
    WEATHER = 'clear'
    FEC_RATE = 0.8
    PILOT_RATIO = 0.1
    N_INFO_BITS = 819 * 8  # Multiple of total k_ldpc = FEC_RATE * 1024 * n_modes
    N_GRID = 512
    OVERSAMPLING = 2
    EQ_METHOD = 'mmse'  # MMSE equalization
    ADD_NOISE = True    # Enable additive noise
    SNR_DB = 35         # Signal-to-noise ratio in dB

    # Adaptive Optics Configuration
    ENABLE_AO = False   # Enable adaptive optics correction
    AO_METHOD = 'modal' # Correction method: 'modal' (fast) or 'sensorless' (slower, better)
    AO_N_ZERNIKE = 15  # Number of Zernike modes for modal correction

    PLOT_DIR = os.path.join(SCRIPT_DIR, "e2e_results_ideal")
    DPI = 1200
    ENABLE_POWER_PROBE = True


def run_e2e_simulation(config, verbose=True):
    """
    Runs the complete, rectified E2E simulation.

    Args:
        config: SimulationConfig instance
        verbose: If True, print progress messages. If False, suppress output.

    Returns:
        Dictionary with simulation results and metrics
    """


    cfg = config
    n_modes = len(cfg.SPATIAL_MODES)

    # 1a. Initialize Transmitter
    if verbose:
        print("[1] Initializing Transmitter...")
    transmitter = encodingRunner(
        spatial_modes=cfg.SPATIAL_MODES,
        wavelength=cfg.WAVELENGTH,
        w0=cfg.W0,
        fec_rate=cfg.FEC_RATE,
        pilot_ratio=cfg.PILOT_RATIO
    )

    # 1b. Initialize Turbulence Model
    print("[2] Initializing Channel Models...")
    turbulence = AtmosphericTurbulence(
        Cn2=cfg.CN2, L0=cfg.L0, l0=cfg.L0_INNER, wavelength=cfg.WAVELENGTH
    )

    # 1c. Initialize Simulation Grid
    print("[3] Initializing Simulation Grid...")
    max_m2_beam = max(transmitter.lg_beams.values(), key=lambda b: b.M_squared)
    beam_size_at_rx = max_m2_beam.beam_waist(cfg.DISTANCE)

    D = cfg.OVERSAMPLING * 6 * beam_size_at_rx
    delta = D / cfg.N_GRID

    x = np.linspace(-D/2, D/2, cfg.N_GRID)
    y = np.linspace(-D/2, D/2, cfg.N_GRID)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    grid_info = {
        'x': x, 'y': y, 'X': X, 'Y': Y, 'R': R, 'PHI': PHI,
        'D': D, 'delta': delta, 'N': cfg.N_GRID
    }
    print(f"    Grid sized for: LG_p={max_m2_beam.p}, l={max_m2_beam.l} (M²={max_m2_beam.M_squared:.1f})")
    print(f"    Grid resolution: {cfg.N_GRID}x{cfg.N_GRID}, Pixel: {delta*1000:.2f} mm")

    # 1d. Generate Basis Fields (at z=0) – FIXED: Init dict + scale for total P_tx
    print("[4] Generating Basis Mode Fields (at z=0)...")
    dA = delta**2
    tx_basis_fields = {}  # FIXED: Define dict here
    basis_energy = {}  # Per-mode energy for scaling
    basis_scaling_factors = {}  # CRITICAL: Store scaling factors for RX reference fields
    for mode_key, beam in transmitter.lg_beams.items():
        E_basis = beam.generate_beam_field(R, PHI, 0)
        energy = np.sum(np.abs(E_basis)**2) * dA  # Unit symbol energy ~1
        basis_energy[mode_key] = energy
        # Scale basis so each mode contrib P_tx / n_modes when symbol=1
        scale = np.sqrt(cfg.P_TX_TOTAL_W / (n_modes * energy))
        basis_scaling_factors[mode_key] = scale  # Store for receiver
        tx_basis_fields[mode_key] = E_basis * scale  # Now |sum basis *1|^2 dA = P_tx
    # FIXED: Validate total power
    E_tx_check = np.sum([tx_basis_fields[mode_key] for mode_key in cfg.SPATIAL_MODES], axis=0)
    total_power = np.sum(np.abs(E_tx_check)**2) * dA
    print(f"    Basis scaled for total TX power: {cfg.P_TX_TOTAL_W} W ({n_modes} modes) – verified: {total_power:.3f} W")
    print(f"    Scaling factors stored for RX reference field matching")

    # 1e. Initialize Receiver
    print("[5] Initializing Receiver...")
    receiver = FSORx(
        spatial_modes=cfg.SPATIAL_MODES,
        wavelength=cfg.WAVELENGTH,
        w0=cfg.W0,
        z_distance=cfg.DISTANCE,
        pilot_handler=transmitter.pilot_handler,  # Share pilot handler!
        eq_method=cfg.EQ_METHOD,
        receiver_radius=(cfg.RECEIVER_DIAMETER / 2.0),
        ldpc_instance=transmitter.ldpc  # CRITICAL: Share LDPC instance to ensure same H matrix!
    )


    # Generate original data bits
    data_bits = np.random.randint(0, 2, cfg.N_INFO_BITS)
    print(f"Generated {len(data_bits)} info bits.")

    # Generate the full frame of symbols
    tx_frame = transmitter.transmit(data_bits, verbose=True)
    tx_signals = tx_frame.tx_signals  # Extract dict from FSO_MDM_Frame object

    # CRITICAL FIX: Set grid_info in tx_frame (receiver needs it for demultiplexing)
    tx_frame.grid_info = grid_info
    # CRITICAL FIX: Store basis scaling factors in tx_frame metadata for RX reference field scaling
    if not hasattr(tx_frame, 'metadata') or tx_frame.metadata is None:
        tx_frame.metadata = {}
    tx_frame.metadata['basis_scaling_factors'] = basis_scaling_factors
    tx_frame.metadata['basis_energy'] = basis_energy
    # CRITICAL FIX: Store noise flag for receiver (Fix 1: Noise Variance)
    tx_frame.metadata['noise_disabled'] = not cfg.ADD_NOISE
    # Store attenuation factor (will be set after calculation, see below)

    # Get total number of symbols. Find the *minimum* length across all modes.
    symbol_lengths = [len(sig['symbols']) for sig in tx_signals.values()]  # FIXED: Use len(symbols)
    if not symbol_lengths or min(symbol_lengths) == 0:
         print("✗ ERROR: Transmitter produced 0 symbols.")
         return None

    n_symbols = min(symbol_lengths)
    # FIXED: Get pilot positions from tx_frame (PilotHandler doesn't have pilot_positions attribute)
    # Pilot positions are stored per-mode in tx_signals[mode_key]["pilot_positions"]
    first_mode = list(tx_signals.keys())[0]
    pilot_pos = np.asarray(tx_signals[first_mode].get("pilot_positions", []), dtype=int)
    if n_symbols < len(pilot_pos):
        print(f"✗ ERROR: n_symbols={n_symbols} < pilots={len(pilot_pos)}. Increase N_INFO_BITS.")
        return None
    print(f"    (Simulation will truncate to minimum frame length: {n_symbols} symbols)")
    print(f"    Pilot positions: {len(pilot_pos)} pilots per mode")



    # 3a. Create one "frozen" snapshot of the atmosphere
    print(f"[1] Generating {cfg.NUM_SCREENS} phase screens for one channel snapshot...")
    layers = create_multi_layer_screens(
        cfg.DISTANCE, cfg.NUM_SCREENS,
        cfg.WAVELENGTH, cfg.CN2,
        cfg.L0, cfg.L0_INNER,
        cn2_model=getattr(cfg, "CN2_MODEL", "hufnagel_valley"),
        verbose=False
    )
    print(f"    Generated {len(layers)} screen layers.")

    # 3b. Calculate Attenuation Loss
    print("[2] Calculating Attenuation...")
    receiver_radius = cfg.RECEIVER_DIAMETER / 2.0
    per_mode_eta = {}
    per_mode_geo_db = {}
    eta_sum = 0.0
    for mode_key, beam in transmitter.lg_beams.items():
        L_geo_dB_mode, eta_mode = calculate_geometric_loss(beam, cfg.DISTANCE, receiver_radius)
        per_mode_eta[mode_key] = eta_mode
        per_mode_geo_db[mode_key] = L_geo_dB_mode
        eta_sum += eta_mode
    eta_weighted = eta_sum / max(1, len(per_mode_eta))
    best_mode = max(per_mode_eta, key=per_mode_eta.get)
    worst_mode = min(per_mode_eta, key=per_mode_eta.get)

    # Use 23km visibility for 'clear' (Kim model)
    visibility_km = 23.0
    alpha_dBkm = calculate_kim_attenuation(cfg.WAVELENGTH * 1e9, visibility_km)
    L_atm_dB = alpha_dBkm * (cfg.DISTANCE / 1000.0)

    amplitude_loss = 10**(-L_atm_dB / 20.0) # Apply attenuation
    coll_eff = eta_weighted * (amplitude_loss ** 2)  # Total collection efficiency
    P_rx_expected = cfg.P_TX_TOTAL_W * coll_eff
    print(f"    Atmospheric Loss: {L_atm_dB:.2f} dB (Amplitude factor: {amplitude_loss:.3f})")
    avg_geo_loss_db = -10 * np.log10(max(eta_weighted, 1e-12))
    print(f"    Geometric Loss (mode-weighted): {avg_geo_loss_db:.2f} dB (Avg Eff: {eta_weighted*100:.2f}%)")
    print(f"      Best mode {best_mode}: {per_mode_eta[best_mode]*100:.1f}%   Worst mode {worst_mode}: {per_mode_eta[worst_mode]*100:.1f}%")
    print(f"    Total P_rx expected: {P_rx_expected:.3f} W (eff incl. atm: {coll_eff*100:.2f}%)")

    # CRITICAL FIX: Store attenuation factor in tx_frame metadata for RX reference field matching
    if not hasattr(tx_frame, 'metadata') or tx_frame.metadata is None:
        tx_frame.metadata = {}
    tx_frame.metadata['amplitude_loss'] = amplitude_loss
    tx_frame.metadata['mode_collection_eff'] = per_mode_eta.copy()

    # Aperture mask & area (reuse later)
    aperture_mask = (grid_info['R'] <= cfg.RECEIVER_DIAMETER / 2.0).astype(float)
    dA = grid_info['delta']**2
    
    # DEBUG: Verify aperture mask statistics
    n_pixels_in_aperture = np.sum(aperture_mask)
    n_pixels_total = aperture_mask.size
    aperture_fraction = n_pixels_in_aperture / n_pixels_total
    print(f"    [DEBUG] Aperture mask: {n_pixels_in_aperture}/{n_pixels_total} pixels ({aperture_fraction*100:.2f}%)")
    print(f"    [DEBUG] Receiver radius: {cfg.RECEIVER_DIAMETER/2.0:.3f} m, Grid extent: {grid_info['D']/2:.3f} m")
    print(f"    [DEBUG] Mask min/max: {np.min(aperture_mask):.1f}/{np.max(aperture_mask):.1f}, unique values: {len(np.unique(aperture_mask))}")
    
    # DEBUG: Calculate beam waists at receiver
    print(f"    [DEBUG] Beam waists at {cfg.DISTANCE}m:")
    for mode_key in cfg.SPATIAL_MODES:
        beam = transmitter.lg_beams[mode_key]
        w = beam.beam_waist(cfg.DISTANCE)
        inside_aperture = "INSIDE" if w < cfg.RECEIVER_DIAMETER/2.0 else "OUTSIDE"
        print(f"      Mode {mode_key}: w = {w*1000:.1f} mm ({inside_aperture} 250mm aperture)")

    if getattr(cfg, "ENABLE_POWER_PROBE", True):
        # Numerical probe: propagate unit symbol frame (all modes = 1) through same channel
        E_tx_probe = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for mode_key in cfg.SPATIAL_MODES:
            E_tx_probe += tx_basis_fields[mode_key]
        with redirect_stdout(io.StringIO()):
            probe_result = apply_multi_layer_turbulence(
                initial_field=E_tx_probe,
                base_beam=max_m2_beam,
                layers=layers,
                total_distance=cfg.DISTANCE,
                N=cfg.N_GRID,
                oversampling=cfg.OVERSAMPLING,
                L0=cfg.L0,
                l0=cfg.L0_INNER,
            )
        E_rx_probe_before_mask = probe_result['final_field'] * amplitude_loss
        P_rx_before_mask = np.sum(np.abs(E_rx_probe_before_mask) ** 2) * dA
        
        # DEBUG: Analyze radial power distribution
        field_power = np.abs(E_rx_probe_before_mask) ** 2
        print(f"    [DEBUG] Radial power distribution:")
        for r_max in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            mask_r = (grid_info['R'] <= r_max)
            power_r = np.sum(field_power * mask_r) * dA
            pct = power_r / P_rx_before_mask * 100 if P_rx_before_mask > 0 else 0
            marker = " ← APERTURE" if abs(r_max - 0.25) < 0.01 else ""
            print(f"      r ≤ {r_max:.2f}m: {power_r:.3f} W ({pct:.1f}%){marker}")
        
        E_rx_probe = E_rx_probe_before_mask * aperture_mask
        P_rx_numeric = np.sum(np.abs(E_rx_probe) ** 2) * dA
        tx_frame.metadata['numeric_power_probe_W'] = float(P_rx_numeric)
        print(f"    [DEBUG] P_rx before aperture: {P_rx_before_mask:.3f} W")
        print(f"    [DEBUG] P_rx after aperture: {P_rx_numeric:.3f} W (reduction: {(1-P_rx_numeric/P_rx_before_mask)*100:.1f}%)")
        print(f"    Numerical P_rx (unit symbol sum): {P_rx_numeric:.3f} W (eff: {P_rx_numeric / cfg.P_TX_TOTAL_W * 100:.2f}%)")
    else:
        print("    Numerical P_rx probe: skipped (ENABLE_POWER_PROBE=False)")

    # 3c. Calculate Noise – FIXED: Per-symbol power probe
    print("[3] Calculating Noise Parameters...")
    num_pixels_in_aperture = np.sum(aperture_mask)
    if num_pixels_in_aperture == 0: num_pixels_in_aperture = 1

    if cfg.ADD_NOISE:
        # Probe: One multiplexed symbol (all modes=1)
        E_tx_probe = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for mode_key in cfg.SPATIAL_MODES:
            E_tx_probe += tx_basis_fields[mode_key]  # Scaled basis *1
        # Suppress verbose turbulence output
        with redirect_stdout(io.StringIO()):
            result_probe = apply_multi_layer_turbulence(
                initial_field=E_tx_probe,
                base_beam=max_m2_beam, layers=layers, total_distance=cfg.DISTANCE,
                N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING,
                L0=cfg.L0, l0=cfg.L0_INNER
            )
        E_rx_probe = result_probe['final_field'] * amplitude_loss * aperture_mask

        power_per_symbol = np.sum(np.abs(E_rx_probe)**2) * dA  # Total for one symbol
        avg_pixel_intensity = power_per_symbol / num_pixels_in_aperture

        snr_linear = 10**(cfg.SNR_DB / 10.0)
        noise_var_per_pixel = avg_pixel_intensity / snr_linear
        noise_std_per_pixel = np.sqrt(noise_var_per_pixel)
        
        # CRITICAL FIX: Store true noise variance in metadata for receiver
        tx_frame.metadata['noise_var_per_pixel'] = float(noise_var_per_pixel)
        tx_frame.metadata['snr_db'] = float(cfg.SNR_DB)

        # FIXED: Validate P_rx from probe
        P_rx_probe = np.sum(np.abs(E_rx_probe)**2) * dA
        print(f"    Target SNR: {cfg.SNR_DB} dB")
        print(f"    Power per Symbol (in aperture): {power_per_symbol:.2e} W")
        print(f"    Avg. Signal Intensity (per pixel): {avg_pixel_intensity:.2e}")
        print(f"    Noise Variance (per pixel): {noise_var_per_pixel:.2e}")
        print(f"    Probe P_rx: {P_rx_probe:.3f} W (matches expected: {P_rx_expected:.3f})")
    else:
        print("    Noise disabled.")
        noise_std_per_pixel = 0.0
        tx_frame.metadata['noise_var_per_pixel'] = 0.0
        tx_frame.metadata['snr_db'] = float('inf')

    # 3d. Loop over all symbols (PHYSICAL PROPAGATION)
    print(f"[4] Propagating {n_symbols} symbols through channel...")
    
    # AO metrics tracking
    ao_metrics_all = [] if getattr(cfg, 'ENABLE_AO', False) and _HAS_AO else None

    E_rx_sequence = [] # This will store the list of 2D fields

    # Sample symbols for TX vis (average first 5 non-pilot or unit) – FIXED: Realistic avg
    sample_syms = np.ones(n_modes, dtype=complex)  # Fallback unit
    first_non_pilot = max(0, len(pilot_pos)) if len(pilot_pos) > 0 else 0
    if first_non_pilot < n_symbols:
        n_samples = min(5, n_symbols - first_non_pilot)
        if n_samples > 0:
            for i in range(n_samples):
                idx = first_non_pilot + i
                for j, mode_key in enumerate(cfg.SPATIAL_MODES):
                    sample_syms[j] += tx_signals[mode_key]['symbols'][idx]
            sample_syms /= (n_samples + 1)  # Avg (including initial ones)

    # Use tqdm for progress bar, suppress turbulence verbose output
    for sym_idx in tqdm(range(n_symbols), desc="Propagating symbols", unit="symbol"): # Loop to the *minimum* length
        # 1. Create the multiplexed field for this symbol – uses scaled basis
        E_tx_symbol = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for i, mode_key in enumerate(cfg.SPATIAL_MODES):
            symbol = tx_signals[mode_key]['symbols'][sym_idx]
            E_tx_symbol += tx_basis_fields[mode_key] * symbol

        # 2. Propagate the *combined* field (suppress verbose output)
        with redirect_stdout(io.StringIO()):
            result = apply_multi_layer_turbulence(
                initial_field=E_tx_symbol,
                base_beam=max_m2_beam, layers=layers, total_distance=cfg.DISTANCE,
                N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING,
                L0=cfg.L0, l0=cfg.L0_INNER
            )
        E_rx_turbulent = result['final_field']

        # 2.5. Apply Adaptive Optics Correction (if enabled)
        ao_metrics_symbol = None
        if getattr(cfg, 'ENABLE_AO', False) and _HAS_AO:
            try:
                # Pass pristine field and phase screens for accurate aberration estimation
                grid_info_with_refs = grid_info.copy()
                grid_info_with_refs['pristine_field'] = result.get('pristine_field', None)
                grid_info_with_refs['phase_screens'] = result.get('phase_screens', None)
                
                E_rx_turbulent, ao_metrics_symbol = apply_adaptive_optics(
                    E_rx_turbulent, grid_info_with_refs, cfg.SPATIAL_MODES,
                    method=getattr(cfg, 'AO_METHOD', 'modal'),
                    n_zernike_modes=getattr(cfg, 'AO_N_ZERNIKE', 15)
                )
                if ao_metrics_all is not None and ao_metrics_symbol is not None:
                    ao_metrics_all.append(ao_metrics_symbol)
            except Exception as e:
                if verbose and sym_idx == 0:  # Only warn once
                    warnings.warn(f"AO correction failed: {e}")
                ao_metrics_symbol = None

        # 3. Apply Attenuation
        E_rx_attenuated = E_rx_turbulent * amplitude_loss

        # 4. Add Noise
        if cfg.ADD_NOISE:
            noise = (noise_std_per_pixel / np.sqrt(2)) * (
                np.random.randn(cfg.N_GRID, cfg.N_GRID) +
                1j * np.random.randn(cfg.N_GRID, cfg.N_GRID)
            )
            E_rx_final = E_rx_attenuated + noise
        else:
            E_rx_final = E_rx_attenuated

        # 5. Apply Aperture (at the very end)
        E_rx_final = E_rx_final * aperture_mask

        # 6. Store the final field
        E_rx_sequence.append(E_rx_final)

    print("    ✓ Full frame propagated.")

    # FIXED: TX vis as multiplexed sample
    E_tx_visualization = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
    for i, mode_key in enumerate(cfg.SPATIAL_MODES):
        E_tx_visualization += tx_basis_fields[mode_key] * sample_syms[i]
    E_rx_visualization = E_rx_sequence[0]


    # Pass the *entire sequence of fields* to the receiver
    # NOTE: tx_frame.grid_info is now set (line 183), so receiver can use it directly
    # The receive_sequence() wrapper will use tx_frame if provided, or construct from grid_info+tx_signals
    recovered_bits, metrics = receiver.receive_sequence(
        E_rx_sequence=E_rx_sequence,
        tx_frame=tx_frame,  # Pass complete tx_frame (includes grid_info and tx_signals)
        original_data_bits=data_bits,  # INFO bits (before LDPC encoding) for BER calculation
        verbose=True
    )


    print(f"    TURBULENCE: Cn² = {cfg.CN2:.2e} (m^-2/3)")
    print(f"    LINK: {cfg.DISTANCE} m, {cfg.NUM_SCREENS} screens")
    print(f"    SNR: {cfg.SNR_DB} dB")
    print(f"    EQUALIZER: {cfg.EQ_METHOD.upper()}")
    if getattr(cfg, 'ENABLE_AO', False) and _HAS_AO and ao_metrics_all:
        # Aggregate AO metrics
        avg_purity_before = np.mean([m.get('purity_before', 0) for m in ao_metrics_all])
        avg_purity_after = np.mean([m.get('purity_after', 0) for m in ao_metrics_all])
        avg_cond_before = np.mean([m.get('cond_before', np.inf) for m in ao_metrics_all])
        avg_cond_after = np.mean([m.get('cond_after', np.inf) for m in ao_metrics_all])
        print(f"    ADAPTIVE OPTICS: {getattr(cfg, 'AO_METHOD', 'modal').upper()}")
        print(f"      Mode Purity: {avg_purity_before:.4f} → {avg_purity_after:.4f} "
              f"(+{(avg_purity_after-avg_purity_before)*100:.2f}%)")
        print(f"      Cond(H): {avg_cond_before:.2e} → {avg_cond_after:.2e}")
    print(f"    TOTAL INFO BITS: {metrics['total_bits']}")
    print(f"    BIT ERRORS:      {metrics['bit_errors']}")
    print(f"    FINAL BER:       {metrics['ber']:.4e}")


    # Store results for plotting
    results = {
        'config': cfg,
        'metrics': metrics,
        'grid_info': grid_info,
        'tx_signals': tx_signals,
        'E_tx_visualization': E_tx_visualization,
        'E_rx_visualization': E_rx_visualization,
        'H_est': metrics['H_est']
    }
    
    # Add AO metrics if available
    if ao_metrics_all:
        results['ao_metrics'] = ao_metrics_all
        
        # Use receiver's H_est for accurate crosstalk measurement
        # H_est is computed from actual pilot symbols and is more accurate than projection-based method
        H_est_from_receiver = metrics.get('H_est', None)
        
        # Compute aggregate statistics
        ao_summary = {
            # Existing metrics
            'avg_purity_before': float(np.mean([m.get('purity_before', 0) for m in ao_metrics_all])),
            'avg_purity_after': float(np.mean([m.get('purity_after', 0) for m in ao_metrics_all])),
            'avg_cond_before': float(np.mean([m.get('cond_before', np.inf) for m in ao_metrics_all])),
            'avg_cond_after': float(np.mean([m.get('cond_after', np.inf) for m in ao_metrics_all])),
            # New comprehensive metrics
            'avg_coupling_efficiency_before': float(np.mean([m.get('coupling_efficiency_before', 0) for m in ao_metrics_all])),
            'avg_coupling_efficiency_after': float(np.mean([m.get('coupling_efficiency_after', 0) for m in ao_metrics_all])),
            'avg_power_loss_dB': float(np.mean([m.get('power_loss_dB', 0) for m in ao_metrics_all if m.get('power_loss_dB') is not None])),
            'avg_phase_error_rms_before': float(np.mean([m.get('phase_error_rms_before', 0) for m in ao_metrics_all if m.get('phase_error_rms_before') is not None])),
            'avg_phase_error_rms_after': float(np.mean([m.get('phase_error_rms_after', 0) for m in ao_metrics_all if m.get('phase_error_rms_after') is not None])),
            'avg_phase_error_reduction_rad': float(np.mean([m.get('phase_error_reduction_rad', 0) for m in ao_metrics_all if m.get('phase_error_reduction_rad') is not None])),
            'method': ao_metrics_all[0].get('method', 'unknown') if ao_metrics_all else 'none'
        }
        
        # Use H_est from receiver for accurate crosstalk measurement
        if H_est_from_receiver is not None:
            from adaptiveOptics import ModePuritySensor
            sensor = ModePuritySensor(cfg.SPATIAL_MODES, cfg.WAVELENGTH, cfg.W0, cfg.DISTANCE)
            
            # Compute comprehensive metrics from H_est (more accurate than projection-based)
            crosstalk_dB_hest = sensor.compute_crosstalk_dB(H_est_from_receiver)
            coupling_efficiency_hest = sensor.compute_coupling_efficiency(H_est_from_receiver)
            cond_H_est = sensor.compute_condition_metric(H_est_from_receiver)
            purity_H_est = sensor.compute_purity_metric(H_est_from_receiver)
            
            ao_summary['crosstalk_dB_H_est'] = float(crosstalk_dB_hest) if np.isfinite(crosstalk_dB_hest) else None
            ao_summary['coupling_efficiency_H_est'] = float(coupling_efficiency_hest)
            ao_summary['cond_H_est'] = float(cond_H_est) if np.isfinite(cond_H_est) else None
            ao_summary['purity_H_est'] = float(purity_H_est)
            ao_summary['H_est'] = H_est_from_receiver
            
            # Also compute from projection-based method for comparison
            if len(ao_metrics_all) > 0 and 'H_after' in ao_metrics_all[0]:
                # Use the last symbol's H_after as representative
                H_projection = ao_metrics_all[-1].get('H_after', None)
                if H_projection is not None:
                    crosstalk_dB_proj = sensor.compute_crosstalk_dB(H_projection)
                    ao_summary['crosstalk_dB_projection'] = float(crosstalk_dB_proj) if np.isfinite(crosstalk_dB_proj) else None
        else:
            # Fallback to projection-based method if H_est not available
            ao_summary['avg_crosstalk_dB_before'] = float(np.mean([m.get('crosstalk_dB_before', np.inf) for m in ao_metrics_all if m.get('crosstalk_dB_before') is not None and np.isfinite(m.get('crosstalk_dB_before'))]))
            ao_summary['avg_crosstalk_dB_after'] = float(np.mean([m.get('crosstalk_dB_after', np.inf) for m in ao_metrics_all if m.get('crosstalk_dB_after') is not None and np.isfinite(m.get('crosstalk_dB_after'))]))
            ao_summary['avg_crosstalk_reduction_dB'] = float(np.mean([m.get('crosstalk_reduction_dB', 0) for m in ao_metrics_all if m.get('crosstalk_reduction_dB') is not None and np.isfinite(m.get('crosstalk_reduction_dB'))]))
        
        results['ao_summary'] = ao_summary
        
        # Print comprehensive metrics if available
        ao_summary = results.get('ao_summary', {})
        if 'avg_coupling_efficiency_before' in ao_summary:
            ce_before = ao_summary['avg_coupling_efficiency_before']
            ce_after = ao_summary['avg_coupling_efficiency_after']
            print(f"      Coupling Efficiency: {ce_before:.4f} → {ce_after:.4f} "
                  f"(+{(ce_after-ce_before)*100:.2f}%)")
        
        # Display crosstalk metrics (prefer H_est if available)
        if 'crosstalk_dB_H_est' in ao_summary and ao_summary['crosstalk_dB_H_est'] is not None:
            xt_hest = ao_summary['crosstalk_dB_H_est']
            print(f"      Crosstalk (H_est): {xt_hest:.2f} dB")
            if 'crosstalk_dB_projection' in ao_summary and ao_summary['crosstalk_dB_projection'] is not None:
                xt_proj = ao_summary['crosstalk_dB_projection']
                print(f"      Crosstalk (projection): {xt_proj:.2f} dB (for comparison)")
            if 'cond_H_est' in ao_summary and ao_summary['cond_H_est'] is not None:
                cond_hest = ao_summary['cond_H_est']
                print(f"      Cond(H_est): {cond_hest:.2e}")
            if 'coupling_efficiency_H_est' in ao_summary:
                ce_hest = ao_summary['coupling_efficiency_H_est']
                print(f"      Coupling Efficiency (H_est): {ce_hest:.4f}")
        elif 'avg_crosstalk_reduction_dB' in ao_summary and np.isfinite(ao_summary.get('avg_crosstalk_reduction_dB', np.nan)):
            xt_reduction = ao_summary['avg_crosstalk_reduction_dB']
            print(f"      Crosstalk Reduction: {xt_reduction:+.2f} dB")
        
        if 'avg_phase_error_reduction_rad' in ao_summary and ao_summary.get('avg_phase_error_reduction_rad') is not None:
            phase_reduction = ao_summary['avg_phase_error_reduction_rad']
            print(f"      Phase Error Reduction: {phase_reduction:.4f} rad (RMS)")

    return results

def plot_e2e_results(results, save_path=None):
    """
    Plot the summary of the E2E simulation.
    """
    print("Generating E2E results plot...")

    cfg = results['config']
    metrics = results['metrics']
    grid_info = results['grid_info']
    H_est = metrics['H_est']

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 3)

    fig.suptitle(f"End-to-End FSO-OAM Simulation Results\n"
                 f"Cn²={cfg.CN2:.1e}, L={cfg.DISTANCE}m, SNR={cfg.SNR_DB}dB, BER={metrics['ber']:.2e}",
                 fontsize=18, fontweight='bold')

    extent_mm = grid_info['D'] * 1e3 / 2

    # FIXED: TX vis as multiplexed
    ax1 = fig.add_subplot(gs[0, 0])
    E_tx_vis = np.abs(results['E_tx_visualization'])**2
    im1 = ax1.imshow(E_tx_vis.T,
                    extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                    cmap='hot', origin='lower')
    ax1.set_title('TX Multiplexed Field Example', fontweight='bold')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    plt.colorbar(im1, ax=ax1, label='Intensity [W/m²]')

    # Plot 2: Received field (example snapshot)
    ax2 = fig.add_subplot(gs[1, 0])
    E_rx_vis = np.abs(results['E_rx_visualization'])**2
    vmax = np.percentile(E_rx_vis, 99.9) # Clip hotspots for better viz
    im2 = ax2.imshow(E_rx_vis.T,
                    extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                    cmap='hot', origin='lower', vmax=vmax)
    ax2.set_title(f'RX Field Snapshot (Symbol 0)', fontweight='bold')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    plt.colorbar(im2, ax=ax2, label='Intensity [W/m²]')

    # Plot 3: Estimated Channel Matrix |H_est|
    ax3 = fig.add_subplot(gs[0, 1])
    im3 = ax3.imshow(np.abs(H_est), cmap='viridis', interpolation='nearest')
    ax3.set_title(r'Estimated Channel Matrix $|\hat{H}|$', fontweight='bold')

    mode_labels = [f"({p},{l})" for p,l in cfg.SPATIAL_MODES]
    ax3.set_xticks(np.arange(len(mode_labels)))
    ax3.set_yticks(np.arange(len(mode_labels)))
    ax3.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax3.set_yticklabels(mode_labels)
    ax3.set_xlabel('Transmitted Mode (j)')
    ax3.set_ylabel('Received Mode (i)')
    plt.colorbar(im3, ax=ax3, label='Magnitude (Coupling Strength)')
    # Add text labels
    for i in range(H_est.shape[0]):
        for j in range(H_est.shape[1]):
            ax3.text(j, i, f"{np.abs(H_est[i,j]):.2f}",
                     ha="center", va="center", color="w", fontsize=8)

    # Plot 4: Channel Matrix Phase
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.angle(H_est), cmap='hsv', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    ax4.set_title(r'Estimated Channel Matrix Phase $\angle \hat{H}$', fontweight='bold')
    ax4.set_xticks(np.arange(len(mode_labels)))
    ax4.set_yticks(np.arange(len(mode_labels)))
    ax4.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax4.set_yticklabels(mode_labels)
    ax4.set_xlabel('Transmitted Mode (j)')
    ax4.set_ylabel('Received Mode (i)')
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')

    # Plot 5: Performance Metrics Text
#     ax5 = fig.add_subplot(gs[:, 2])
#     ax5.axis('off')

#     # Get turbulence properties
#     temp_turb = AtmosphericTurbulence(
#         Cn2=cfg.CN2, L0=cfg.L0, l0=cfg.L0_INNER, wavelength=cfg.WAVELENGTH
#     )

#     metrics_text = f"""
# SYSTEM PERFORMANCE METRICS

# [Link Parameters]
#   Distance: {cfg.DISTANCE} m
#   Weather: {cfg.WEATHER}
#   Turbulence: Cn² = {cfg.CN2:.2e}
#   SNR: {cfg.SNR_DB} dB
#   Modes: {len(cfg.SPATIAL_MODES)} ( {', '.join(mode_labels)} )

# [Channel Metrics]
#   Rytov Variance: {temp_turb.rytov_variance(cfg.DISTANCE):.3f}
#   Fried Parameter (r0): {temp_turb.fried_parameter(cfg.DISTANCE)*1000:.2f} mm
#   Channel Condition: {np.linalg.cond(H_est):.2f}

# [Receiver Metrics]
#   Equalization: {cfg.EQ_METHOD.upper()}
#   Est. Noise Var: {metrics['noise_var']:.2e}

# [FINAL PERFORMANCE]
#   Total Info Bits: {metrics['total_bits']}
#   Bit Errors: {metrics['bit_errors']}
#   ---------------------------------
#   Bit Error Rate (BER): {metrics['ber']:.4e}
#   ---------------------------------
#     """

#     ax5.text(0.0, 0.95, metrics_text, transform=ax5.transAxes,
#             fontsize=12, verticalalignment='top', fontfamily='monospace',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if save_path:
#         # Ensure the directory exists before trying to save the file
#         plot_directory = os.path.dirname(save_path)
#         os.makedirs(plot_directory, exist_ok=True)

#         plt.savefig(save_path, dpi=cfg.DPI, bbox_inches='tight')
#         print(f"\n✓ E2E Results plot saved to: {save_path}")

    return fig


def plot_symbol_comparison(results, save_path=None, max_points=512):
    """
    Plot transmitted vs received constellations per spatial mode.
    """
    metrics = results.get("metrics", {})
    cfg = results.get("config")
    tx_samples = metrics.get("tx_symbols_sample")
    rx_samples = metrics.get("rx_symbols_sample")
    if rx_samples is None:
        print("Symbol comparison plot skipped: receiver metrics missing rx_symbols_sample.")
        return None

    modes = cfg.SPATIAL_MODES if cfg is not None else list(range(rx_samples.shape[0]))
    n_modes = len(modes)
    rows = int(np.ceil(n_modes / 3))
    cols = min(n_modes, 3)
    figsize = (6 * cols, 6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
    fig.suptitle("Transmitted vs Received Constellations per Mode", fontsize=18, fontweight="bold")

    for idx, mode_key in enumerate(modes):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        rx = rx_samples[idx]
        if max_points is not None and max_points > 0:
            rx = rx[:max_points]
        ax.scatter(rx.real, rx.imag, s=10, alpha=0.5, color="#d97706", label="RX (post-eq)")
        if tx_samples is not None:
            tx = tx_samples[idx]
            if max_points is not None and max_points > 0:
                tx = tx[:max_points]
            ax.scatter(tx.real, tx.imag, s=10, alpha=0.5, color="#1d4ed8", label="TX (pre-channel)")
        ax.set_title(f"Mode {mode_key}", fontweight="bold")
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9)

    # Hide unused axes if any
    total_axes = rows * cols
    for idx in range(n_modes, total_axes):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    if save_path:
        plot_directory = os.path.dirname(save_path)
        os.makedirs(plot_directory, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.DPI if cfg is not None else 1200, bbox_inches="tight")
        print(f"✓ Symbol comparison plot saved to: {save_path}")
    return fig


def run_cn2_sweep(config_class, cn2_values, enable_power_probe=False, save_plots=False):
    """
    Sweep through a list of Cn² values and record BER / conditioning metrics.
    """


    summary = []
    for idx, cn2 in enumerate(cn2_values, start=1):
        cfg = config_class()
        cfg.CN2 = cn2
        cfg.ENABLE_POWER_PROBE = enable_power_probe

        print(f"\n--- Sweep {idx}/{len(cn2_values)}: CN² = {cn2:.2e} ---")
        results = run_e2e_simulation(cfg)
        metrics = results['metrics']
        cond_h = metrics.get("cond_H", float(np.linalg.cond(metrics['H_est'])))
        coded_ber = metrics.get("coded_ber")

        summary.append({
            "cn2": cn2,
            "ber": metrics['ber'],
            "bit_errors": metrics['bit_errors'],
            "cond_H": cond_h,
            "coded_ber": coded_ber
        })

        # Optional plot export per sweep step
        if save_plots:
            sweep_dir = os.path.join(cfg.PLOT_DIR, "cn2_sweep")
            os.makedirs(sweep_dir, exist_ok=True)
            plot_path = os.path.join(sweep_dir, f"cn2_{cn2:.2e}.png")
            plot_e2e_results(results, save_path=plot_path)
            symbol_path = os.path.join(sweep_dir, f"cn2_{cn2:.2e}_symbols.png")
            plot_symbol_comparison(results, save_path=symbol_path)

    print("\nSweep Summary:")
    print(f"{'CN² (m^-2/3)':>14} | {'BER':>10} | {'Bit Err':>8} | {'cond(H)':>10} | {'Coded BER':>10}")
    print("-"*70)
    for row in summary:
        coded_str = f"{row['coded_ber']:.3e}" if row['coded_ber'] is not None else "n/a"
        print(f"{row['cn2']:.2e} | {row['ber']:.3e} | {row['bit_errors']:8d} | {row['cond_H']:.3e} | {coded_str:>10}")

    return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FSO-OAM End-to-End Simulation")
    parser.add_argument(
        "--cn2-sweep",
        nargs="+",
        type=float,
        help="List of Cn² values (m^-2/3) to sweep. Example: --cn2-sweep 0 5e-19 1e-18"
    )
    parser.add_argument(
        "--disable-power-probe",
        action="store_true",
        help="Skip the numerical power probe diagnostic to speed up runs."
    )
    parser.add_argument(
        "--save-sweep-plots",
        action="store_true",
        help="When sweeping Cn², save plots for each operating point."
    )
    args = parser.parse_args()

    if args.cn2_sweep:
        cn2_values = args.cn2_sweep
        run_cn2_sweep(
            SimulationConfig,
            cn2_values,
            enable_power_probe=not args.disable_power_probe,
            save_plots=args.save_sweep_plots
        )
    else:
        # Single-run path
        config = SimulationConfig()
        if args.disable_power_probe:
            config.ENABLE_POWER_PROBE = False

        results = run_e2e_simulation(config)
        if results:
            save_file = os.path.join(config.PLOT_DIR, "e2e_simulation_results.png")
            fig = plot_e2e_results(results, save_path=save_file)
            symbol_file = os.path.join(config.PLOT_DIR, "e2e_symbol_comparison.png")
            symbol_fig = plot_symbol_comparison(results, save_path=symbol_file)
            plt.show()
        else:
            print("✗ Simulation failed to produce results.")
