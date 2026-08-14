#!/usr/bin/env python3
"""
interactive_main.py
Ask the user for a NACA 4-digit airfoil, drive XFOIL automatically to
generate comparison data, then run the full thin-airfoil-theory pipeline
(flow field, polar, chordwise loading) against it.

Run this from the same directory as your XFOIL executable.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from config import (X_MIN, X_MAX, Y_MIN, Y_MAX, NX, NY, C, V_INF, N_FOURIER)
from fourier_coefficients import fourier_coefficients
from vortex import circulation, lift_coefficient
from velocity import compute_field_grid
from visualize import plot_pressure_field, plot_streamfunction_heatmap, y_camber
from generate_polar import compute_polar
from generate_loading import compute_loading
from xfoil_runner import run_full_case, parse_naca4


# ------------------------------------------------------------
# Small input helpers
# ------------------------------------------------------------
def ask(prompt, default, cast=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  Could not parse '{raw}', using default {default}.")
        return default


def ask_naca_code():
    while True:
        code = input("NACA 4-digit airfoil (e.g. 2412): ").strip()
        try:
            geom = parse_naca4(code)
            return geom
        except ValueError as e:
            print(f"  {e} -- try again.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== Thin Airfoil Theory vs XFOIL -- interactive run ===\n")

    geom = ask_naca_code()
    naca, M, P, T = geom['naca'], geom['M'], geom['P'], geom['T']
    print(f"  -> M={M:.3f}, P={P:.3f}, T={T:.3f} (thickness is used by XFOIL geometry "
          f"only; thin-airfoil theory here is a camber-line theory and ignores thickness)\n")

    alpha_show_deg = ask("Angle of attack to visualize (flow field & loading), deg", 5.0, float)
    alpha_min_deg = ask("Polar sweep: alpha min (deg)", -5.0, float)
    alpha_max_deg = ask("Polar sweep: alpha max (deg)", 15.0, float)
    alpha_step_deg = ask("Polar sweep: alpha step (deg)", 1.0, float)
    xfoil_exe = ask("XFOIL executable name/path", "xfoil.exe", str)
    iter_limit = ask("XFOIL iteration limit (ITER)", 400, int)

    visc_choice = ask("Run XFOIL viscous? (y/n) -- 'n' matches the inviscid thin-airfoil theory", "n", str)
    viscous_re = None
    if visc_choice.lower().startswith("y"):
        viscous_re = ask("  Reynolds number", 1_000_000, float)

    print()
    xfoil_result = run_full_case(
        naca, alpha_min=alpha_min_deg, alpha_max=alpha_max_deg, alpha_step=alpha_step_deg,
        alpha_cp=alpha_show_deg, xfoil_exe=xfoil_exe, work_dir=".",
        iter_limit=iter_limit, viscous_re=viscous_re,
    )
    polar_file = xfoil_result['polar_file']
    cpwr_file = xfoil_result['cpwr_file']

    ALPHA = np.deg2rad(alpha_show_deg)

    # ------------------------------------------------------------
    # 1. Flow field visualization
    # ------------------------------------------------------------
    print("\n--- Flow field computation ---")
    A = fourier_coefficients(M, P, ALPHA, N_FOURIER)
    print(f"A0 = {A[0]:.6f}, A1 = {A[1]:.6f}, A2 = {A[2]:.6f}")

    cl = lift_coefficient(A)
    Gamma = circulation(V_INF, C, A)
    print(f"Lift coefficient: c_l = {cl:.6f}")
    print(f"Circulation: Gamma = {Gamma:.6f}")

    X = np.linspace(X_MIN, X_MAX, NX)
    Y = np.linspace(Y_MIN, Y_MAX, NY)
    X_grid, Y_grid = np.meshgrid(X, Y)

    field = compute_field_grid(X_grid, Y_grid, V_INF, ALPHA, C, A,
                                include_pressure=True, include_velocity=True,
                                epsrel=1e-6)
    U, V, Cp = field['U'], field['V'], field['Cp']

    x_camber = np.linspace(0, C, 200)
    y_camber_vals = y_camber(x_camber, C, M, P) if M > 0 else np.zeros_like(x_camber)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_pressure_field(X_grid, Y_grid, Cp, ax=ax1)
    ax1.set_title(r'Pressure Coefficient $C_p$')
    ax1.plot(x_camber, y_camber_vals, 'k-', linewidth=1.5)
    ax1.plot([0, C], [0, 0], 'k--', alpha=0.4, linewidth=1)
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_ylim(Y_MIN, Y_MAX)

    plot_streamfunction_heatmap(X_grid, Y_grid, U, V, ax=ax2, mask_radius=0.05)
    ax2.set_title(r'Streamlines (Stream Function $\psi$)')
    ax2.plot(x_camber, y_camber_vals, 'k-', linewidth=1.5)
    ax2.plot([0, C], [0, 0], 'k--', alpha=0.4, linewidth=1)
    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_ylim(Y_MIN, Y_MAX)

    plt.tight_layout()
    flow_field_png = f"flow_field_{naca}.png"
    plt.savefig(flow_field_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Flow field plot saved to {flow_field_png}")

    # ------------------------------------------------------------
    # 2. Polar comparison (Cl vs alpha)
    # ------------------------------------------------------------
    print("\n--- Polar comparison ---")
    polar_result = compute_polar(
        alpha_min_deg=alpha_min_deg, alpha_max_deg=alpha_max_deg, alpha_step_deg=alpha_step_deg,
        xfoil_polar_file=polar_file, plot=True, m=M, p=P, naca_label=naca,
    )

    # ------------------------------------------------------------
    # 3. Chordwise loading comparison (Delta Cp vs x/c)
    # ------------------------------------------------------------
    print("\n--- Loading comparison ---")
    loading_result = compute_loading(
        alpha_show_deg, xfoil_cpwr_file=cpwr_file, plot=True, m=M, p=P, naca_label=naca,
    )

    # ------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"NACA {naca}")
    print(f"Thin-airfoil Cl at alpha = {alpha_show_deg:.1f} deg: {cl:.6f}")
    print(f"Flow field plot:   {flow_field_png}")
    print(f"Polar comparison:  {polar_result['comparison_png']}")
    print(f"Loading comparison saved.")
    print("All done!")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt,):
        print("\nCancelled.")
        sys.exit(1)