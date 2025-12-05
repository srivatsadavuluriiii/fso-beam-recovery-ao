"""
Cn² Sweep: Compare performance with and without AO across turbulence levels.

Sweeps Cn² from 10e-18 to 10e-12 and generates comprehensive comparison plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

from pipeline import SimulationConfig, run_e2e_simulation
from adaptiveOptics import ModePuritySensor

warnings.filterwarnings('ignore')


def sweep_cn2_with_ao_comparison(cn2_values=None, n_zernike=35, n_info_bits=200*8, 
                                 n_grid=256, verbose=False):
    """
    Sweep Cn² values and compare performance with/without AO.
    
    Args:
        cn2_values: List of Cn² values to sweep (default: logspace from 10e-18 to 10e-12)
        n_zernike: Number of Zernike modes for AO
        n_info_bits: Number of info bits (smaller = faster)
        n_grid: Grid size (smaller = faster)
        verbose: Print detailed progress
    
    Returns:
        Dictionary with sweep results
    """
    if cn2_values is None:
        # Default: logspace from 10e-18 to 10e-12
        cn2_values = np.logspace(-18, -12, 7, base=10)
    
    print("="*70)
    print("CN² SWEEP: AO COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Cn² range: {cn2_values[0]:.1e} to {cn2_values[-1]:.1e} m⁻²/³")
    print(f"  Number of points: {len(cn2_values)}")
    print(f"  Zernike modes: {n_zernike}")
    print(f"  Grid size: {n_grid}x{n_grid}")
    print()
    
    # Storage for results
    results = {
        'cn2_values': cn2_values,
        'no_ao': {
            'ber': [],
            'crosstalk_dB': [],
            'coupling_efficiency': [],
            'mode_purity': [],
            'condition_number': [],
            'results': []
        },
        'with_ao': {
            'ber': [],
            'crosstalk_dB': [],
            'coupling_efficiency': [],
            'mode_purity': [],
            'condition_number': [],
            'results': []
        }
    }
    
    # Initialize sensor for metric computation
    config_temp = SimulationConfig()
    sensor = ModePuritySensor(config_temp.SPATIAL_MODES, config_temp.WAVELENGTH,
                             config_temp.W0, config_temp.DISTANCE)
    
    # Sweep through Cn² values
    for idx, cn2 in enumerate(tqdm(cn2_values, desc="Sweeping Cn²")):
        if verbose:
            print(f"\n[{idx+1}/{len(cn2_values)}] Cn² = {cn2:.1e} m⁻²/³")
            print("-" * 70)
        
        # Base configuration
        config = SimulationConfig()
        config.CN2 = cn2
        config.N_INFO_BITS = n_info_bits
        config.N_GRID = n_grid
        config.ADD_NOISE = True
        config.ENABLE_POWER_PROBE = False
        
        # Run WITHOUT AO
        if verbose:
            print("  Running WITHOUT AO...")
        config_no_ao = config
        config_no_ao.ENABLE_AO = False
        
        try:
            result_no_ao = run_e2e_simulation(config_no_ao, verbose=False)
            if result_no_ao is None:
                if verbose:
                    print("    ✗ Failed")
                continue
            
            H_no_ao = result_no_ao['metrics']['H_est']
            results['no_ao']['ber'].append(result_no_ao['metrics']['ber'])
            results['no_ao']['crosstalk_dB'].append(sensor.compute_crosstalk_dB(H_no_ao))
            results['no_ao']['coupling_efficiency'].append(sensor.compute_coupling_efficiency(H_no_ao))
            results['no_ao']['mode_purity'].append(sensor.compute_purity_metric(H_no_ao))
            results['no_ao']['condition_number'].append(sensor.compute_condition_metric(H_no_ao))
            results['no_ao']['results'].append(result_no_ao)
            
            if verbose:
                print(f"    ✓ BER: {result_no_ao['metrics']['ber']:.4e}")
        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue
        
        # Run WITH AO
        if verbose:
            print("  Running WITH AO...")
        config_ao = config
        config_ao.ENABLE_AO = True
        config_ao.AO_METHOD = 'modal'
        config_ao.AO_N_ZERNIKE = n_zernike
        
        try:
            result_ao = run_e2e_simulation(config_ao, verbose=False)
            if result_ao is None:
                if verbose:
                    print("    ✗ Failed")
                continue
            
            H_ao = result_ao['metrics']['H_est']
            results['with_ao']['ber'].append(result_ao['metrics']['ber'])
            results['with_ao']['crosstalk_dB'].append(sensor.compute_crosstalk_dB(H_ao))
            results['with_ao']['coupling_efficiency'].append(sensor.compute_coupling_efficiency(H_ao))
            results['with_ao']['mode_purity'].append(sensor.compute_purity_metric(H_ao))
            results['with_ao']['condition_number'].append(sensor.compute_condition_metric(H_ao))
            results['with_ao']['results'].append(result_ao)
            
            if verbose:
                print(f"    ✓ BER: {result_ao['metrics']['ber']:.4e}")
        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue
    
    # Convert to numpy arrays
    for key in ['ber', 'crosstalk_dB', 'coupling_efficiency', 'mode_purity', 'condition_number']:
        results['no_ao'][key] = np.array(results['no_ao'][key])
        results['with_ao'][key] = np.array(results['with_ao'][key])
    
    print("\n✓ Sweep completed!")
    return results


def plot_cn2_sweep_comparison(sweep_results, save_path=None, dpi=1200):
    """
    Plot comprehensive comparison of Cn² sweep results.
    
    Args:
        sweep_results: Results from sweep_cn2_with_ao_comparison()
        save_path: Path to save figure
        dpi: Resolution
    """
    cn2_values = sweep_results['cn2_values']
    no_ao = sweep_results['no_ao']
    with_ao = sweep_results['with_ao']
    
    # Filter out invalid values
    valid_idx = np.isfinite(no_ao['ber']) & np.isfinite(with_ao['ber'])
    cn2_valid = cn2_values[valid_idx]
    
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: BER vs Cn²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(cn2_valid, no_ao['ber'][valid_idx], 'o-', 
                 label='Without AO', color='#d97706', linewidth=2, markersize=8)
    ax1.semilogx(cn2_valid, with_ao['ber'][valid_idx], 's-',
                 label='With AO', color='#1d4ed8', linewidth=2, markersize=8)
    ax1.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax1.set_ylabel('BER', fontsize=11)
    ax1.set_title('Bit Error Rate vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Crosstalk vs Cn²
    ax2 = fig.add_subplot(gs[0, 1])
    valid_xt = np.isfinite(no_ao['crosstalk_dB']) & np.isfinite(with_ao['crosstalk_dB'])
    cn2_xt = cn2_values[valid_xt]
    ax2.semilogx(cn2_xt, no_ao['crosstalk_dB'][valid_xt], 'o-',
                 label='Without AO', color='#d97706', linewidth=2, markersize=8)
    ax2.semilogx(cn2_xt, with_ao['crosstalk_dB'][valid_xt], 's-',
                 label='With AO', color='#1d4ed8', linewidth=2, markersize=8)
    ax2.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax2.set_ylabel('Crosstalk [dB]', fontsize=11)
    ax2.set_title('Crosstalk vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Coupling Efficiency vs Cn²
    ax3 = fig.add_subplot(gs[1, 0])
    valid_ce = np.isfinite(no_ao['coupling_efficiency']) & np.isfinite(with_ao['coupling_efficiency'])
    cn2_ce = cn2_values[valid_ce]
    ax3.semilogx(cn2_ce, no_ao['coupling_efficiency'][valid_ce], 'o-',
                 label='Without AO', color='#d97706', linewidth=2, markersize=8)
    ax3.semilogx(cn2_ce, with_ao['coupling_efficiency'][valid_ce], 's-',
                 label='With AO', color='#1d4ed8', linewidth=2, markersize=8)
    ax3.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax3.set_ylabel('Coupling Efficiency', fontsize=11)
    ax3.set_title('Coupling Efficiency vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Mode Purity vs Cn²
    ax4 = fig.add_subplot(gs[1, 1])
    valid_purity = np.isfinite(no_ao['mode_purity']) & np.isfinite(with_ao['mode_purity'])
    cn2_purity = cn2_values[valid_purity]
    ax4.semilogx(cn2_purity, no_ao['mode_purity'][valid_purity], 'o-',
                 label='Without AO', color='#d97706', linewidth=2, markersize=8)
    ax4.semilogx(cn2_purity, with_ao['mode_purity'][valid_purity], 's-',
                 label='With AO', color='#1d4ed8', linewidth=2, markersize=8)
    ax4.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax4.set_ylabel('Mode Purity', fontsize=11)
    ax4.set_title('Mode Purity vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1])
    
    # Plot 5: Condition Number vs Cn²
    ax5 = fig.add_subplot(gs[2, 0])
    valid_cond = np.isfinite(no_ao['condition_number']) & np.isfinite(with_ao['condition_number'])
    cn2_cond = cn2_values[valid_cond]
    ax5.semilogx(cn2_cond, no_ao['condition_number'][valid_cond], 'o-',
                 label='Without AO', color='#d97706', linewidth=2, markersize=8)
    ax5.semilogx(cn2_cond, with_ao['condition_number'][valid_cond], 's-',
                 label='With AO', color='#1d4ed8', linewidth=2, markersize=8)
    ax5.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax5.set_ylabel('Condition Number', fontsize=11)
    ax5.set_title('Channel Condition Number vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    
    # Plot 6: Improvement vs Cn²
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Calculate improvements
    ber_improvement = (no_ao['ber'][valid_idx] - with_ao['ber'][valid_idx]) / no_ao['ber'][valid_idx] * 100
    crosstalk_reduction = no_ao['crosstalk_dB'][valid_xt] - with_ao['crosstalk_dB'][valid_xt]
    coupling_improvement = (with_ao['coupling_efficiency'][valid_ce] - no_ao['coupling_efficiency'][valid_ce]) / no_ao['coupling_efficiency'][valid_ce] * 100
    
    ax6_twin = ax6.twinx()
    
    # BER improvement (left axis)
    line1 = ax6.semilogx(cn2_valid, ber_improvement, 'o-', 
                         label='BER Improvement [%]', color='green', linewidth=2, markersize=8)
    ax6.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax6.set_ylabel('BER Improvement [%]', fontsize=11, color='green')
    ax6.tick_params(axis='y', labelcolor='green')
    
    # Crosstalk reduction (right axis)
    line2 = ax6_twin.semilogx(cn2_xt, crosstalk_reduction, 's-',
                              label='Crosstalk Reduction [dB]', color='blue', linewidth=2, markersize=8)
    ax6_twin.set_ylabel('Crosstalk Reduction [dB]', fontsize=11, color='blue')
    ax6_twin.tick_params(axis='y', labelcolor='blue')
    
    ax6.set_title('AO Improvement vs Turbulence Strength', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left', fontsize=9)
    
    fig.suptitle('Adaptive Optics Performance Across Turbulence Levels\n'
                 f'Cn² Sweep: {cn2_values[0]:.1e} to {cn2_values[-1]:.1e} m⁻²/³',
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved sweep comparison to: {save_path}")
    
    return fig


def plot_improvement_summary(sweep_results, save_path=None, dpi=1200):
    """
    Plot summary of AO improvements across turbulence levels.
    
    Args:
        sweep_results: Results from sweep_cn2_with_ao_comparison()
        save_path: Path to save figure
        dpi: Resolution
    """
    cn2_values = sweep_results['cn2_values']
    no_ao = sweep_results['no_ao']
    with_ao = sweep_results['with_ao']
    
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Calculate improvements
    valid_idx = np.isfinite(no_ao['ber']) & np.isfinite(with_ao['ber'])
    cn2_valid = cn2_values[valid_idx]
    
    ber_improvement = (no_ao['ber'][valid_idx] - with_ao['ber'][valid_idx]) / no_ao['ber'][valid_idx] * 100
    crosstalk_reduction = no_ao['crosstalk_dB'][valid_idx] - with_ao['crosstalk_dB'][valid_idx]
    coupling_improvement = (with_ao['coupling_efficiency'][valid_idx] - no_ao['coupling_efficiency'][valid_idx]) / no_ao['coupling_efficiency'][valid_idx] * 100
    purity_improvement = (with_ao['mode_purity'][valid_idx] - no_ao['mode_purity'][valid_idx]) / no_ao['mode_purity'][valid_idx] * 100
    
    # Plot 1: BER Improvement
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(cn2_valid, ber_improvement, 'o-', color='green', linewidth=2, markersize=8)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax1.set_ylabel('BER Improvement [%]', fontsize=11)
    ax1.set_title('BER Improvement', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(cn2_valid, 0, ber_improvement, where=(ber_improvement > 0), 
                     alpha=0.3, color='green', label='Improvement')
    ax1.fill_between(cn2_valid, 0, ber_improvement, where=(ber_improvement < 0), 
                     alpha=0.3, color='red', label='Degradation')
    ax1.legend(fontsize=9)
    
    # Plot 2: Crosstalk Reduction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(cn2_valid, crosstalk_reduction, 's-', color='blue', linewidth=2, markersize=8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax2.set_ylabel('Crosstalk Reduction [dB]', fontsize=11)
    ax2.set_title('Crosstalk Reduction', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(cn2_valid, 0, crosstalk_reduction, where=(crosstalk_reduction > 0), 
                     alpha=0.3, color='blue', label='Reduction')
    ax2.fill_between(cn2_valid, 0, crosstalk_reduction, where=(crosstalk_reduction < 0), 
                     alpha=0.3, color='red', label='Increase')
    ax2.legend(fontsize=9)
    
    # Plot 3: Coupling Efficiency Improvement
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogx(cn2_valid, coupling_improvement, '^-', color='purple', linewidth=2, markersize=8)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax3.set_ylabel('Coupling Efficiency Improvement [%]', fontsize=11)
    ax3.set_title('Coupling Efficiency Improvement', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(cn2_valid, 0, coupling_improvement, where=(coupling_improvement > 0), 
                     alpha=0.3, color='purple')
    
    # Plot 4: Mode Purity Improvement
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogx(cn2_valid, purity_improvement, 'd-', color='orange', linewidth=2, markersize=8)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Cn² [m⁻²/³]', fontsize=11)
    ax4.set_ylabel('Mode Purity Improvement [%]', fontsize=11)
    ax4.set_title('Mode Purity Improvement', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(cn2_valid, 0, purity_improvement, where=(purity_improvement > 0), 
                     alpha=0.3, color='orange')
    
    fig.suptitle('Adaptive Optics Improvement Summary\n'
                 f'Across Turbulence Levels (Cn²: {cn2_values[0]:.1e} to {cn2_values[-1]:.1e} m⁻²/³)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved improvement summary to: {save_path}")
    
    return fig


def run_cn2_sweep_analysis(cn2_start=10e-18, cn2_end=10e-12, n_points=7, 
                           n_zernike=35, output_dir='plots/cn2_sweep'):
    """
    Complete workflow: Run sweep and generate all visualizations.
    
    Args:
        cn2_start: Starting Cn² value
        cn2_end: Ending Cn² value
        n_points: Number of points in sweep
        n_zernike: Number of Zernike modes
        output_dir: Output directory
    """
    # Generate Cn² values
    cn2_values = np.logspace(np.log10(cn2_start), np.log10(cn2_end), n_points)
    
    print("="*70)
    print("CN² SWEEP ANALYSIS: AO COMPARISON")
    print("="*70)
    print(f"\nSweep Parameters:")
    print(f"  Cn² range: {cn2_start:.1e} to {cn2_end:.1e} m⁻²/³")
    print(f"  Number of points: {n_points}")
    print(f"  Zernike modes: {n_zernike}")
    print()
    
    # Run sweep
    results = sweep_cn2_with_ao_comparison(
        cn2_values=cn2_values,
        n_zernike=n_zernike,
        n_info_bits=200*8,  # Smaller for faster runs
        n_grid=256,  # Smaller grid for faster runs
        verbose=False
    )
    
    # Generate visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Main comparison plot
    print("\n[1/2] Generating main comparison plot...")
    fig1 = plot_cn2_sweep_comparison(
        results,
        save_path=os.path.join(output_dir, 'cn2_sweep_comparison.png')
    )
    plt.close(fig1)
    
    # Improvement summary
    print("\n[2/2] Generating improvement summary...")
    fig2 = plot_improvement_summary(
        results,
        save_path=os.path.join(output_dir, 'cn2_sweep_improvements.png')
    )
    plt.close(fig2)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    valid_idx = np.isfinite(results['no_ao']['ber']) & np.isfinite(results['with_ao']['ber'])
    if np.sum(valid_idx) > 0:
        ber_improvement = (results['no_ao']['ber'][valid_idx] - results['with_ao']['ber'][valid_idx]) / results['no_ao']['ber'][valid_idx] * 100
        crosstalk_reduction = results['no_ao']['crosstalk_dB'][valid_idx] - results['with_ao']['crosstalk_dB'][valid_idx]
        
        print(f"\nAverage Improvements:")
        print(f"  BER improvement: {np.mean(ber_improvement):.1f}% (range: {np.min(ber_improvement):.1f}% to {np.max(ber_improvement):.1f}%)")
        print(f"  Crosstalk reduction: {np.mean(crosstalk_reduction):.2f} dB (range: {np.min(crosstalk_reduction):.2f} to {np.max(crosstalk_reduction):.2f} dB)")
        
        print(f"\nBest Performance (strongest turbulence):")
        best_idx = np.argmax(cn2_values[valid_idx])
        print(f"  Cn² = {cn2_values[valid_idx][best_idx]:.1e} m⁻²/³")
        print(f"  BER improvement: {ber_improvement[best_idx]:.1f}%")
        print(f"  Crosstalk reduction: {crosstalk_reduction[best_idx]:.2f} dB")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cn² sweep with AO comparison")
    parser.add_argument("--cn2-start", type=float, default=10e-18,
                       help="Starting Cn² value (default: 10e-18)")
    parser.add_argument("--cn2-end", type=float, default=10e-12,
                       help="Ending Cn² value (default: 10e-12)")
    parser.add_argument("--n-points", type=int, default=7,
                       help="Number of points in sweep (default: 7)")
    parser.add_argument("--n-zernike", type=int, default=35,
                       help="Number of Zernike modes (default: 35)")
    parser.add_argument("--output-dir", type=str, default="plots/cn2_sweep",
                       help="Output directory (default: plots/cn2_sweep)")
    
    args = parser.parse_args()
    
    run_cn2_sweep_analysis(
        cn2_start=args.cn2_start,
        cn2_end=args.cn2_end,
        n_points=args.n_points,
        n_zernike=args.n_zernike,
        output_dir=args.output_dir
    )

