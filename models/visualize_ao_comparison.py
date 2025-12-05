"""
Comprehensive visualization module for Adaptive Optics before/after comparisons.

Creates publication-quality plots showing:
1. Field intensity and phase before/after AO
2. Crosstalk matrices before/after
3. Performance metrics comparison
4. Zernike coefficient analysis
5. Mode purity improvements
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, TwoSlopeNorm
import warnings

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

warnings.filterwarnings('ignore')


def plot_ao_field_comparison(results_no_ao, results_ao, save_path=None, dpi=1200):
    """
    Compare field intensity and phase before/after AO.
    
    Args:
        results_no_ao: Simulation results without AO
        results_ao: Simulation results with AO
        save_path: Path to save figure
        dpi: Resolution for saved figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    cfg = results_ao['config']
    grid_info = results_ao['grid_info']
    extent_mm = grid_info['D'] * 1e3 / 2
    
    # Get fields
    E_rx_no_ao = results_no_ao['E_rx_visualization']
    E_rx_ao = results_ao['E_rx_visualization']
    
    I_no_ao = np.abs(E_rx_no_ao)**2
    I_ao = np.abs(E_rx_ao)**2
    phase_no_ao = np.angle(E_rx_no_ao)
    phase_ao = np.angle(E_rx_ao)
    
    # Row 1: Intensity
    # Before AO
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(I_no_ao, 99.9)
    im1 = ax1.imshow(I_no_ao.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='hot', origin='lower', vmax=vmax)
    ax1.set_title('Intensity: Before AO', fontweight='bold', fontsize=12)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    plt.colorbar(im1, ax=ax1, label='Intensity [W/m²]')
    
    # After AO
    ax2 = fig.add_subplot(gs[0, 1])
    vmax_ao = np.percentile(I_ao, 99.9)
    im2 = ax2.imshow(I_ao.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='hot', origin='lower', vmax=vmax_ao)
    ax2.set_title('Intensity: After AO', fontweight='bold', fontsize=12)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    plt.colorbar(im2, ax=ax2, label='Intensity [W/m²]')
    
    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    I_diff = I_ao - I_no_ao
    vmax_diff = np.max(np.abs(I_diff))
    im3 = ax3.imshow(I_diff.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='RdBu_r', origin='lower', vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_title('Intensity Difference (After - Before)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    plt.colorbar(im3, ax=ax3, label='Δ Intensity [W/m²]')
    
    # Row 2: Phase
    # Before AO
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(phase_no_ao.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase: Before AO', fontweight='bold', fontsize=12)
    ax4.set_xlabel('x [mm]')
    ax4.set_ylabel('y [mm]')
    plt.colorbar(im4, ax=ax4, label='Phase [rad]')
    
    # After AO
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(phase_ao.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax5.set_title('Phase: After AO', fontweight='bold', fontsize=12)
    ax5.set_xlabel('x [mm]')
    ax5.set_ylabel('y [mm]')
    plt.colorbar(im5, ax=ax5, label='Phase [rad]')
    
    # Phase Error Reduction
    ax6 = fig.add_subplot(gs[1, 2])
    # Compute phase error (if pristine field available)
    if 'pristine_field' in results_ao.get('grid_info', {}):
        E_pristine = results_ao['grid_info']['pristine_field']
        phase_pristine = np.angle(E_pristine)
        phase_error_before = phase_no_ao - phase_pristine
        phase_error_after = phase_ao - phase_pristine
        phase_error_diff = phase_error_after - phase_error_before
    else:
        # Use phase difference directly
        phase_error_diff = phase_ao - phase_no_ao
    
    # Unwrap and normalize
    phase_error_diff = np.angle(np.exp(1j * phase_error_diff))
    im6 = ax6.imshow(phase_error_diff.T, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                     cmap='RdBu_r', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax6.set_title('Phase Error Reduction', fontweight='bold', fontsize=12)
    ax6.set_xlabel('x [mm]')
    ax6.set_ylabel('y [mm]')
    plt.colorbar(im6, ax=ax6, label='Δ Phase Error [rad]')
    
    fig.suptitle(f'Adaptive Optics Field Comparison\n'
                 f'Cn²={cfg.CN2:.1e} m⁻²/³, L={cfg.DISTANCE}m, {cfg.AO_N_ZERNIKE} Zernike modes',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved field comparison to: {save_path}")
    
    return fig


def plot_ao_crosstalk_comparison(results_no_ao, results_ao, save_path=None, dpi=1200):
    """
    Compare crosstalk matrices before/after AO.
    
    Args:
        results_no_ao: Simulation results without AO
        results_ao: Simulation results with AO
        save_path: Path to save figure
        dpi: Resolution for saved figure
    """
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)
    
    cfg = results_ao['config']
    H_no_ao = results_no_ao['metrics']['H_est']
    H_ao = results_ao['metrics']['H_est']
    
    mode_labels = [f"({p},{l})" for p, l in cfg.SPATIAL_MODES]
    
    # Before AO
    ax1 = fig.add_subplot(gs[0, 0])
    H_abs_no_ao = np.abs(H_no_ao)
    im1 = ax1.imshow(H_abs_no_ao, cmap='viridis', interpolation='nearest')
    ax1.set_title('Crosstalk Matrix: Before AO', fontweight='bold', fontsize=12)
    ax1.set_xticks(np.arange(len(mode_labels)))
    ax1.set_yticks(np.arange(len(mode_labels)))
    ax1.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax1.set_yticklabels(mode_labels)
    ax1.set_xlabel('Transmitted Mode (j)')
    ax1.set_ylabel('Received Mode (i)')
    plt.colorbar(im1, ax=ax1, label='|H_ij|')
    # Add text labels
    for i in range(H_abs_no_ao.shape[0]):
        for j in range(H_abs_no_ao.shape[1]):
            ax1.text(j, i, f"{H_abs_no_ao[i,j]:.2f}",
                    ha="center", va="center", 
                    color="w" if H_abs_no_ao[i,j] > np.max(H_abs_no_ao)*0.5 else "k",
                    fontsize=8)
    
    # After AO
    ax2 = fig.add_subplot(gs[0, 1])
    H_abs_ao = np.abs(H_ao)
    im2 = ax2.imshow(H_abs_ao, cmap='viridis', interpolation='nearest')
    ax2.set_title('Crosstalk Matrix: After AO', fontweight='bold', fontsize=12)
    ax2.set_xticks(np.arange(len(mode_labels)))
    ax2.set_yticks(np.arange(len(mode_labels)))
    ax2.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax2.set_yticklabels(mode_labels)
    ax2.set_xlabel('Transmitted Mode (j)')
    ax2.set_ylabel('Received Mode (i)')
    plt.colorbar(im2, ax=ax2, label='|H_ij|')
    # Add text labels
    for i in range(H_abs_ao.shape[0]):
        for j in range(H_abs_ao.shape[1]):
            ax2.text(j, i, f"{H_abs_ao[i,j]:.2f}",
                    ha="center", va="center",
                    color="w" if H_abs_ao[i,j] > np.max(H_abs_ao)*0.5 else "k",
                    fontsize=8)
    
    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    H_diff = H_abs_ao - H_abs_no_ao
    vmax_diff = np.max(np.abs(H_diff))
    im3 = ax3.imshow(H_diff, cmap='RdBu_r', interpolation='nearest',
                     vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_title('Crosstalk Reduction (After - Before)', fontweight='bold', fontsize=12)
    ax3.set_xticks(np.arange(len(mode_labels)))
    ax3.set_yticks(np.arange(len(mode_labels)))
    ax3.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax3.set_yticklabels(mode_labels)
    ax3.set_xlabel('Transmitted Mode (j)')
    ax3.set_ylabel('Received Mode (i)')
    plt.colorbar(im3, ax=ax3, label='Δ |H_ij|')
    # Add text labels
    for i in range(H_diff.shape[0]):
        for j in range(H_diff.shape[1]):
            ax3.text(j, i, f"{H_diff[i,j]:+.2f}",
                    ha="center", va="center",
                    color="w" if abs(H_diff[i,j]) > vmax_diff*0.5 else "k",
                    fontsize=8)
    
    fig.suptitle(f'Crosstalk Matrix Comparison\n'
                 f'Diagonal = Signal, Off-diagonal = Crosstalk',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved crosstalk comparison to: {save_path}")
    
    return fig


def plot_ao_metrics_comparison(results_no_ao, results_ao, save_path=None, dpi=1200):
    """
    Compare performance metrics before/after AO.
    
    Args:
        results_no_ao: Simulation results without AO
        results_ao: Simulation results with AO
        save_path: Path to save figure
        dpi: Resolution for saved figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract metrics
    metrics_no_ao = results_no_ao['metrics']
    metrics_ao = results_ao['metrics']
    ao_summary = results_ao.get('ao_summary', {})
    
    # Compute metrics from H_est
    from adaptiveOptics import ModePuritySensor
    cfg = results_ao['config']
    sensor = ModePuritySensor(cfg.SPATIAL_MODES, cfg.WAVELENGTH, cfg.W0, cfg.DISTANCE)
    
    H_no_ao = metrics_no_ao['H_est']
    H_ao = metrics_ao['H_est']
    
    # Calculate metrics
    crosstalk_no_ao = sensor.compute_crosstalk_dB(H_no_ao)
    crosstalk_ao = sensor.compute_crosstalk_dB(H_ao)
    coupling_no_ao = sensor.compute_coupling_efficiency(H_no_ao)
    coupling_ao = sensor.compute_coupling_efficiency(H_ao)
    cond_no_ao = sensor.compute_condition_metric(H_no_ao)
    cond_ao = sensor.compute_condition_metric(H_ao)
    purity_no_ao = sensor.compute_purity_metric(H_no_ao)
    purity_ao = sensor.compute_purity_metric(H_ao)
    
    ber_no_ao = metrics_no_ao['ber']
    ber_ao = metrics_ao['ber']
    
    # Plot 1: Bar chart comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['Crosstalk\n(dB)', 'Coupling\nEfficiency', 'Mode\nPurity', 'BER']
    before_values = [crosstalk_no_ao, coupling_no_ao, purity_no_ao, ber_no_ao]
    after_values = [crosstalk_ao, coupling_ao, purity_ao, ber_ao]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_values, width, label='Before AO', color='#d97706', alpha=0.8)
    bars2 = ax1.bar(x + width/2, after_values, width, label='After AO', color='#1d4ed8', alpha=0.8)
    
    ax1.set_ylabel('Value')
    ax1.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Improvement percentages
    ax2 = fig.add_subplot(gs[0, 1])
    improvements = [
        (crosstalk_no_ao - crosstalk_ao) / abs(crosstalk_no_ao) * 100 if crosstalk_no_ao != 0 else 0,  # Crosstalk reduction %
        (coupling_ao - coupling_no_ao) / coupling_no_ao * 100 if coupling_no_ao > 0 else 0,  # Coupling improvement %
        (purity_ao - purity_no_ao) / purity_no_ao * 100 if purity_no_ao > 0 else 0,  # Purity improvement %
        (ber_no_ao - ber_ao) / ber_no_ao * 100 if ber_no_ao > 0 else 0  # BER improvement %
    ]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(metrics_names, improvements, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Improvement [%]')
    ax2.set_title('Relative Improvement', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 3: Condition number comparison
    ax3 = fig.add_subplot(gs[1, 0])
    cond_data = [cond_no_ao, cond_ao]
    cond_labels = ['Before AO', 'After AO']
    colors_cond = ['#d97706', '#1d4ed8']
    bars = ax3.bar(cond_labels, cond_data, color=colors_cond, alpha=0.8)
    ax3.set_ylabel('Condition Number')
    ax3.set_title('Channel Condition Number', fontweight='bold', fontsize=12)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, cond_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Before AO', 'After AO', 'Improvement'],
        ['Crosstalk (dB)', f'{crosstalk_no_ao:.2f}', f'{crosstalk_ao:.2f}', 
         f'{crosstalk_no_ao - crosstalk_ao:+.2f} dB'],
        ['Coupling Efficiency', f'{coupling_no_ao:.4f}', f'{coupling_ao:.4f}',
         f'{(coupling_ao - coupling_no_ao) / coupling_no_ao * 100:+.1f}%'],
        ['Mode Purity', f'{purity_no_ao:.4f}', f'{purity_ao:.4f}',
         f'{(purity_ao - purity_no_ao) / purity_no_ao * 100:+.1f}%'],
        ['BER', f'{ber_no_ao:.4e}', f'{ber_ao:.4e}',
         f'{(ber_no_ao - ber_ao) / ber_no_ao * 100:+.1f}%'],
        ['Condition Number', f'{cond_no_ao:.2e}', f'{cond_ao:.2e}',
         f'{(cond_no_ao - cond_ao) / cond_no_ao * 100:+.1f}%'],
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Table', fontweight='bold', fontsize=12, pad=20)
    
    fig.suptitle(f'Adaptive Optics Performance Comparison\n'
                 f'Cn²={cfg.CN2:.1e} m⁻²/³, {cfg.AO_N_ZERNIKE} Zernike modes',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved metrics comparison to: {save_path}")
    
    return fig


def plot_zernike_analysis(results_ao, save_path=None, dpi=1200):
    """
    Plot Zernike coefficient analysis from AO correction.
    
    Args:
        results_ao: Simulation results with AO
        save_path: Path to save figure
        dpi: Resolution for saved figure
    """
    ao_metrics = results_ao.get('ao_metrics', [])
    if not ao_metrics:
        print("⚠ No AO metrics found for Zernike analysis")
        return None
    
    # Get Zernike coefficients from last symbol (or average)
    zernike_coeffs = ao_metrics[-1].get('zernike_coeffs', {})
    if not zernike_coeffs:
        print("⚠ No Zernike coefficients found")
        return None
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Plot 1: Zernike coefficient magnitudes
    ax1 = fig.add_subplot(gs[0, 0])
    modes = sorted(zernike_coeffs.keys())
    coeffs = [abs(zernike_coeffs[m]) for m in modes]
    
    mode_names = {
        1: 'Piston', 2: 'Tip', 3: 'Tilt', 4: 'Defocus',
        5: 'Astig 45°', 6: 'Astig 0°', 7: 'Coma y', 8: 'Coma x',
        9: 'Trefoil', 10: 'Spherical'
    }
    labels = [mode_names.get(m, f'Z{m}') for m in modes]
    
    bars = ax1.bar(range(len(modes)), coeffs, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Zernike Mode')
    ax1.set_ylabel('|Coefficient| [rad]')
    ax1.set_title('Zernike Coefficient Magnitudes', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(modes)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, coeffs):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Plot 2: Dominant aberrations (top 5)
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_modes = sorted(zernike_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = sorted_modes[:5]
    
    top_modes = [m[0] for m in top_5]
    top_coeffs = [abs(m[1]) for m in top_5]
    top_labels = [mode_names.get(m, f'Z{m}') for m in top_modes]
    
    bars = ax2.barh(range(len(top_5)), top_coeffs, color='crimson', alpha=0.7)
    ax2.set_yticks(range(len(top_5)))
    ax2.set_yticklabels(top_labels)
    ax2.set_xlabel('|Coefficient| [rad]')
    ax2.set_title('Top 5 Dominant Aberrations', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_coeffs)):
        ax2.text(val, i, f' {val:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    cfg = results_ao['config']
    fig.suptitle(f'Zernike Decomposition Analysis\n'
                 f'{cfg.AO_N_ZERNIKE} modes, Cn²={cfg.CN2:.1e} m⁻²/³',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved Zernike analysis to: {save_path}")
    
    return fig


def create_comprehensive_ao_report(results_no_ao, results_ao, output_dir='plots/ao_comparison'):
    """
    Create comprehensive AO comparison report with all visualizations.
    
    Args:
        results_no_ao: Simulation results without AO
        results_ao: Simulation results with AO
        output_dir: Directory to save all plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Creating Comprehensive AO Comparison Report")
    print("="*70)
    
    # 1. Field comparison
    print("\n[1/4] Generating field comparison...")
    fig1 = plot_ao_field_comparison(
        results_no_ao, results_ao,
        save_path=os.path.join(output_dir, 'ao_field_comparison.png')
    )
    plt.close(fig1)
    
    # 2. Crosstalk comparison
    print("\n[2/4] Generating crosstalk comparison...")
    fig2 = plot_ao_crosstalk_comparison(
        results_no_ao, results_ao,
        save_path=os.path.join(output_dir, 'ao_crosstalk_comparison.png')
    )
    plt.close(fig2)
    
    # 3. Metrics comparison
    print("\n[3/4] Generating metrics comparison...")
    fig3 = plot_ao_metrics_comparison(
        results_no_ao, results_ao,
        save_path=os.path.join(output_dir, 'ao_metrics_comparison.png')
    )
    plt.close(fig3)
    
    # 4. Zernike analysis
    print("\n[4/4] Generating Zernike analysis...")
    fig4 = plot_zernike_analysis(
        results_ao,
        save_path=os.path.join(output_dir, 'ao_zernike_analysis.png')
    )
    if fig4:
        plt.close(fig4)
    
    print("\n" + "="*70)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("="*70)
    
    return output_dir


if __name__ == "__main__":
    # Example usage
    print("AO Visualization Module")
    print("="*50)
    print("\nThis module provides visualization functions for AO comparisons.")
    print("Use create_comprehensive_ao_report() to generate all plots.")
    print("\nExample:")
    print("  from visualize_ao_comparison import create_comprehensive_ao_report")
    print("  create_comprehensive_ao_report(results_no_ao, results_ao)")

