#!/usr/bin/env python3
"""
generate_polar.py
Generate lift coefficient polar for a NACA 4-digit airfoil using thin-airfoil theory,
and overlay with XFOIL polar data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import M, P, N_FOURIER, C
from fourier_coefficients import fourier_coefficients
from vortex import lift_coefficient
from xfoil_parser import read_xfoil_polar

# ------------------------------------------------------------
# Public function for use in main.py / interactive_main.py
# ------------------------------------------------------------
def compute_polar(alpha_min_deg=-5.0, alpha_max_deg=15.0, alpha_step_deg=1.0,
                   xfoil_polar_file=None, plot=True, m=None, p=None,
                   naca_label=None, thin_csv=None, comparison_png=None):
    """
    Compute the thin-airfoil Cl-vs-alpha polar and optionally overlay XFOIL data.

    Parameters
    ----------
    alpha_min_deg, alpha_max_deg, alpha_step_deg : float
        Sweep range in degrees.
    xfoil_polar_file : str or None
        Path to an XFOIL polar file. If None, no XFOIL overlay.
    plot : bool
        If True, generate and save the comparison plot.
    m, p : float or None
        Camber / camber-position overrides. Defaults to config.M / config.P.
    naca_label : str or None
        NACA code string to use in the plot title/filenames.
    thin_csv, comparison_png : str or None
        Output filenames. Defaults chosen from naca_label if not given.

    Returns
    -------
    dict with 'alpha_deg', 'Cl', 'df_xfoil' (or None), 'thin_csv', 'comparison_png'
    """
    m_val = M if m is None else m
    p_val = P if p is None else p
    label = naca_label if naca_label is not None else f'{int(m_val*100):02d}{int(p_val*10):01d}??'

    thin_csv = thin_csv or f"polar_thin_airfoil_{label}.csv"
    comparison_png = comparison_png or f"polar_comparison_{label}.png"

    alpha_deg = np.arange(alpha_min_deg, alpha_max_deg + alpha_step_deg / 2, alpha_step_deg)
    alpha_rad = np.deg2rad(alpha_deg)

    cl_values = []
    for a_rad in alpha_rad:
        A = fourier_coefficients(m_val, p_val, a_rad, N_FOURIER)
        cl_values.append(lift_coefficient(A))
    cl_values = np.array(cl_values)

    df_thin = pd.DataFrame({'alpha_deg': alpha_deg, 'Cl': cl_values})
    df_thin.to_csv(thin_csv, index=False)
    print(f"Thin-airfoil polar saved to {thin_csv}")

    df_xfoil = None
    if xfoil_polar_file is not None:
        try:
            df_xfoil = read_xfoil_polar(xfoil_polar_file)
            print(f"XFOIL polar loaded from {xfoil_polar_file}")
            mask = (df_xfoil['alpha'] >= alpha_min_deg) & (df_xfoil['alpha'] <= alpha_max_deg)
            df_xfoil = df_xfoil[mask]
        except Exception as e:
            print(f"Warning: Could not read XFOIL polar file: {e}")
            df_xfoil = None

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5.625))   # 16:9
        ax.plot(alpha_deg, cl_values, 'r-o', linewidth=2, markersize=6, label='Thin airfoil theory')
        if df_xfoil is not None:
            ax.plot(df_xfoil['alpha'], df_xfoil['CL'], 'b-s', linewidth=2, markersize=6, label='XFOIL')

        ax.set_xlabel('Angle of attack $\\alpha$ (degrees)')
        ax.set_ylabel('Lift coefficient $C_l$')
        ax.set_title(f'NACA {label}  Polar comparison')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.legend()
        ax.set_xticks(np.arange(alpha_min_deg, alpha_max_deg + 1, max(1, int(alpha_step_deg))))
        plt.tight_layout()
        plt.savefig(comparison_png, dpi=300)
        plt.show()
        print(f"Comparison plot saved to {comparison_png}")

    return {
        'alpha_deg': alpha_deg,
        'Cl': cl_values,
        'df_xfoil': df_xfoil,
        'thin_csv': thin_csv,
        'comparison_png': comparison_png,
    }


# ------------------------------------------------------------
# If run as a script, use default settings (unchanged behavior)
# ------------------------------------------------------------
if __name__ == "__main__":
    compute_polar(
        alpha_min_deg=-5.0, alpha_max_deg=15.0, alpha_step_deg=1.0,
        xfoil_polar_file="polar.txt", plot=True,
        thin_csv="polar_thin_airfoil.csv", comparison_png="polar_comparison.png",
    )