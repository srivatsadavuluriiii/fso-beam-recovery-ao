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

def text_to_bits(text: str) -> np.ndarray:
    """
    Convert a UTF-8 string into a numpy array of bits (uint8 {0,1}).
    """
    if not text:
        return np.zeros(0, dtype=np.uint8)
    byte_array = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    return np.unpackbits(byte_array)


def bits_to_text(bits: np.ndarray, bit_length: int) -> str:
    """
    Reconstruct a UTF-8 string from a numpy array of bits.
    Only the first `bit_length` bits are considered (remaining bits ignored).
    """
    bit_length = int(bit_length)
    if bit_length <= 0:
        return ""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        bits = bits.ravel()
    trimmed = bits[:bit_length]
    if bit_length % 8 != 0:
        pad = 8 - (bit_length % 8)
        trimmed = np.concatenate([trimmed, np.zeros(pad, dtype=np.uint8)])
    byte_len = (bit_length + 7) // 8
    byte_array = np.packbits(trimmed)[:byte_len]
    try:
        return byte_array.tobytes().decode('utf-8')
    except UnicodeDecodeError:
        # Fallback: ignore invalid trailing bytes
        return byte_array.tobytes().decode('utf-8', errors='ignore')


# Import all modules
try:
    from lgBeam import LaguerreGaussianBeam
    from encoding import encodingRunner, QPSKModulator, PilotHandler
    from fsplAtmAttenuation import calculate_kim_attenuation, calculate_geometric_loss
    
    from turbulence import (AtmosphericTurbulence, 
                            create_multi_layer_screens,
                            apply_multi_layer_turbulence)
    from receiver import FSORx
except ImportError as e:
    print(f"✗ E2E Simulation Import Error: {e}")
    print("  Please ensure lgBeam.py, encoding.py, fsplAtmAttenuation.py, turbulence.py, and receiver.py are in the same directory.")
    sys.exit(1)

np.random.seed(42)

# ============================================================================
# GLOBAL SYSTEM CONFIGURATION
# ============================================================================
class SimulationConfig:
    """
    Centralized configuration for a realistic FSO-OAM system 
    for research purposes.
    """
    # --- Optical Parameters ---
    WAVELENGTH = 1550e-9  # [m]
    W0 = 25e-3           # [m]
    
    # --- Link Parameters ---
    DISTANCE = 1000      # [m] - Updated to match pipeline.py
    RECEIVER_DIAMETER = 0.5  # [m] - Fixed: was 0.3m, increased to reduce geometric loss for high-order modes
    P_TX_TOTAL_W = 1.0     # [W] – now used for scaling
    
    # --- Spatial Modes ---
    SPATIAL_MODES = [
    (0, -1), (0, 1), (0, -3), (0, 3), (0, -4), (0, 4), (1, -1), (1, 1)]
    
    # --- Turbulence Parameters ---
    CN2 = 1e-17           # [m^(-2/3)] - Updated to match pipeline.py
    L0 = 10.0           # [m]
    L0_INNER = 0.005    # [m]
    NUM_SCREENS = 15    # Updated to match pipeline.py
    CN2_MODEL = "uniform"  # Horizontal path → uniform profile; set "hufnagel_valley" for vertical links
    
    # --- Weather Condition ---
    WEATHER = 'clear'    
    
    # --- Communication Parameters ---
    FEC_RATE = 0.8      # Updated to match pipeline.py
    PILOT_RATIO = 0.1   # Updated to match pipeline.py
    
    # FIXED: Multiple of total k_ldpc = FEC_RATE * 1024 * n_modes
    N_INFO_BITS = 819 * 8  # Placeholder; updated dynamically once LDPC is built
    LDPC_BLOCKS = 4       # Number of LDPC codewords per frame
    MESSAGE = "SRIVATSA"
    
    # --- Simulation Grid ---
    N_GRID = 512        # 512x512 is fast for a sanity check
    OVERSAMPLING = 2    
    
    # --- Receiver Configuration ---
    EQ_METHOD = 'mmse'  # Updated to match pipeline.py (use MMSE by default)
    ADD_NOISE = False   # Disable additive noise
    SNR_DB = 35         # Updated to match pipeline.py
    
    # --- Output ---
    PLOT_DIR = os.path.join(SCRIPT_DIR, "e2e_results_ideal") # New folder
    DPI = 1200  # Updated to match pipeline.py
    ENABLE_POWER_PROBE = True  # Toggle numerical power probe diagnostic

# ============================================================================
# NEW E2E SIMULATION (RECTIFIED)
# ============================================================================

def run_e2e_simulation(config):
    """
    Runs the complete, rectified E2E simulation.
    
    This implementation is physically correct AND compatible with all
    your existing files.
    """
    
    # === 1. INITIALIZATION ===
    print("\n" + "="*80)
    print("INITIALIZING E2E SIMULATION")
    print("="*80)
    
    cfg = config
    n_modes = len(cfg.SPATIAL_MODES)

    # 1a. Initialize Transmitter
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
    ldpc_k = transmitter.ldpc.k
    ldpc_n = transmitter.ldpc.n
    num_ldpc_blocks = int(getattr(cfg, "LDPC_BLOCKS", 4))
    if num_ldpc_blocks <= 0:
        num_ldpc_blocks = 1
    n_info_bits = ldpc_k * num_ldpc_blocks
    if getattr(cfg, "N_INFO_BITS", None) != n_info_bits:
        cfg.N_INFO_BITS = n_info_bits
    print(f"    LDPC block dims: k={ldpc_k}, n={ldpc_n}, blocks/frame={num_ldpc_blocks}, total info bits={n_info_bits}")

    # === 2. TRANSMITTER ===
    print("\n" + "="*80)
    print("STAGE 1: TRANSMITTER")
    print("="*80)
    
    # Generate original data bits (embed deterministic message, pad with random tail)
    message_text = getattr(cfg, "MESSAGE", "")
    message_bits = text_to_bits(message_text)
    message_bit_len = len(message_bits)
    if message_bit_len > n_info_bits:
        raise ValueError(
            f"Message '{message_text}' requires {message_bit_len} bits, "
            f"but available info capacity is only {n_info_bits} bits."
        )
    data_bits = np.random.randint(0, 2, n_info_bits)
    if message_bit_len > 0:
        data_bits[:message_bit_len] = message_bits
    print(f"Generated {len(data_bits)} info bits (message '{message_text}' occupies {message_bit_len} bits).")
    
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
    tx_frame.metadata['message_text'] = message_text
    tx_frame.metadata['message_bit_length'] = message_bit_len
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

    # === 3. PHYSICAL CHANNEL ===
    print("\n" + "="*80)
    print("STAGE 2: PHYSICAL CHANNEL (QUASI-STATIC)")
    print("="*80)
    
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
        E_rx_probe = probe_result['final_field'] * amplitude_loss * aperture_mask
        P_rx_numeric = np.sum(np.abs(E_rx_probe) ** 2) * dA
        tx_frame.metadata['numeric_power_probe_W'] = float(P_rx_numeric)
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

    # 3d. Loop over all symbols (PHYSICAL PROPAGATION)
    print(f"[4] Propagating {n_symbols} symbols through channel...")
    
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

    # === 4. RECEIVER ===
    print("\n" + "="*80)
    print("STAGE 3: DIGITAL RECEIVER")
    print("="*80)
    
    # Pass the *entire sequence of fields* to the receiver
    # NOTE: tx_frame.grid_info is now set (line 183), so receiver can use it directly
    # The receive_sequence() wrapper will use tx_frame if provided, or construct from grid_info+tx_signals
    recovered_bits, metrics = receiver.receive_sequence(
        E_rx_sequence=E_rx_sequence,
        tx_frame=tx_frame,  # Pass complete tx_frame (includes grid_info and tx_signals)
        original_data_bits=data_bits,  # INFO bits (before LDPC encoding) for BER calculation
        verbose=True
    )
    decoded_message = bits_to_text(recovered_bits, message_bit_len)
    print(f"\nRecovered message (first {message_bit_len} bits): '{decoded_message}'")
    print(f"Original message: '{message_text}'")

    # === 5. RESULTS ===
    print("\n" + "="*80)
    print("E2E SIMULATION COMPLETE - FINAL RESULTS")
    print("="*80)
    print(f"    TURBULENCE: Cn² = {cfg.CN2:.2e} (m^-2/3)")
    print(f"    LINK: {cfg.DISTANCE} m, {cfg.NUM_SCREENS} screens")
    print(f"    SNR: {cfg.SNR_DB} dB")
    print(f"    EQUALIZER: {cfg.EQ_METHOD.upper()}")
    print(f"    -----------------------------------")
    print(f"    TOTAL INFO BITS: {metrics['total_bits']}")
    print(f"    BIT ERRORS:      {metrics['bit_errors']}")
    print(f"    FINAL BER:       {metrics['ber']:.4e}")
    print("="*80)
    print(f"    MESSAGE TX: '{message_text}'")
    print(f"    MESSAGE RX: '{decoded_message}'")
    print("="*80)
    
    # Store results for plotting
    results = {
        'config': cfg,
        'metrics': metrics,
        'grid_info': grid_info,
        'tx_signals': tx_signals,
        'E_tx_visualization': E_tx_visualization,
        'E_rx_visualization': E_rx_visualization,
        'H_est': metrics['H_est'],
        'message': {
            'original': message_text,
            'decoded': decoded_message,
            'bit_length': message_bit_len
        }
    }
    
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


def run_cn2_sweep(config_class, cn2_values, enable_power_probe=False, save_plots=False):
    """
    Sweep through a list of Cn² values and record BER / conditioning metrics.
    """
    print("\n" + "="*80)
    print("CN² SWEEP: BEGIN")
    print("="*80)

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

    print("\nSweep Summary:")
    print(f"{'CN² (m^-2/3)':>14} | {'BER':>10} | {'Bit Err':>8} | {'cond(H)':>10} | {'Coded BER':>10}")
    print("-"*70)
    for row in summary:
        coded_str = f"{row['coded_ber']:.3e}" if row['coded_ber'] is not None else "n/a"
        print(f"{row['cn2']:.2e} | {row['ber']:.3e} | {row['bit_errors']:8d} | {row['cond_H']:.3e} | {coded_str:>10}")

    print("\nCN² SWEEP: COMPLETE")
    print("="*80)
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
            plt.show()
        else:
            print("✗ Simulation failed to produce results.")
