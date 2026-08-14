#!/usr/bin/env python3
"""
generate_loading.py
Compute thin-airfoil chordwise loading distribution ΔCp(x)
and compare with XFOIL surface pressure data.
Now supports import as a module via compute_loading().
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

from config import M, P, C, V_INF, N_FOURIER
from fourier_coefficients import fourier_coefficients
from vortex import gamma_theta
from velocity import x_of_theta
from xfoil_parser import read_xfoil_cpwr

# ------------------------------------------------------------
# Public function for use in main.py
# ------------------------------------------------------------
def compute_loading(alpha_deg, xfoil_cpwr_file=None, plot=True, m=None, p=None, naca_label=None):
    """
    Compute thin-airfoil loading and optionally compare with XFOIL CPWR data.
    
    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    xfoil_cpwr_file : str or None
        Path to XFOIL CPWR file. If None, no XFOIL comparison.
    plot : bool
        If True, generate and show the comparison plot.
    m, p : float or None
        Camber / camber-position overrides. Defaults to config.M / config.P
        if not given, so existing callers are unaffected.
    naca_label : str or None
        Optional NACA code string to use in the plot title instead of the
        digits reconstructed from m/p (avoids rounding artifacts).
    
    Returns
    -------
    dict
        Contains:
            - 'x_masked' : array of x/c points
            - 'delta_Cp_masked' : thin-airfoil ΔCp
            - 'delta_Cp_xfoil' : XFOIL ΔCp (or None)
            - 'A' : Fourier coefficients
    """
    m_val = M if m is None else m
    p_val = P if p is None else p

    ALPHA_RAD = np.deg2rad(alpha_deg)
    N_THETA = 500
    MASK_FRACTION = 0.02
    
    # Compute thin-airfoil loading
    A = fourier_coefficients(m_val, p_val, ALPHA_RAD, N_FOURIER)
    
    theta_min = 0.01
    theta_max = np.pi - 0.01
    theta = np.linspace(theta_min, theta_max, N_THETA)
    
    gamma = gamma_theta(theta, V_INF, A)
    x_over_c = x_of_theta(theta, C) / C
    delta_Cp = 2.0 * gamma / V_INF
    
    mask = (x_over_c > MASK_FRACTION) & (x_over_c < (1.0 - MASK_FRACTION))
    x_masked = x_over_c[mask]
    delta_Cp_masked = delta_Cp[mask]
    
    # Read XFOIL data if file provided
    delta_Cp_xfoil = None
    if xfoil_cpwr_file is not None:
        try:
            upper, lower = read_xfoil_cpwr(xfoil_cpwr_file)
            interp_u = interp1d(upper[:,0], upper[:,1], kind='linear', fill_value='extrapolate')
            interp_l = interp1d(lower[:,0], lower[:,1], kind='linear', fill_value='extrapolate')
            Cp_u = interp_u(x_masked)
            Cp_l = interp_l(x_masked)
            delta_Cp_xfoil = Cp_l - Cp_u
        except Exception as e:
            print(f"Warning: Could not read XFOIL CPWR file: {e}")
            delta_Cp_xfoil = None
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(x_masked, delta_Cp_masked, 'r-', linewidth=2.5, label='Thin airfoil theory')
        if delta_Cp_xfoil is not None:
            ax.plot(x_masked, delta_Cp_xfoil, 'b--', linewidth=2.5, label='XFOIL')
        label = naca_label if naca_label is not None else f'{int(m_val*100):02d}{int(p_val*10):01d}??'
        ax.set_xlabel('Chordwise position $x/c$')
        ax.set_ylabel('Loading $\\Delta C_p = C_{p,lower} - C_{p,upper}$')
        ax.set_title(f'NACA {label}  Loading at $\\alpha = {alpha_deg:.0f}^\\circ$')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend()
        # Set y-limits
        ymax = max(abs(delta_Cp_masked.min()), abs(delta_Cp_masked.max()))
        if delta_Cp_xfoil is not None:
            ymax = max(ymax, abs(delta_Cp_xfoil.min()), abs(delta_Cp_xfoil.max()))
        ymax *= 1.1
        ax.set_ylim(-ymax, ymax)
        plt.tight_layout()
        plt.savefig(f'loading_comparison_alpha{alpha_deg:.0f}.png', dpi=300)
        plt.show()
        print(f"Loading plot saved to loading_comparison_alpha{alpha_deg:.0f}.png")
    
    return {
        'x_masked': x_masked,
        'delta_Cp_masked': delta_Cp_masked,
        'delta_Cp_xfoil': delta_Cp_xfoil,
        'A': A,
        'Cl': np.pi * (2*A[0] + A[1])
    }

# ------------------------------------------------------------
# If run as a script, use default settings
# ------------------------------------------------------------
if __name__ == "__main__":
    ALPHA_DEG = 5.0
    XFOIL_CPWR_FILE = "cp_2412_a5.txt"   # adjust as needed
    result = compute_loading(ALPHA_DEG, xfoil_cpwr_file=XFOIL_CPWR_FILE, plot=True)
    print(f"Cl = {result['Cl']:.6f}")