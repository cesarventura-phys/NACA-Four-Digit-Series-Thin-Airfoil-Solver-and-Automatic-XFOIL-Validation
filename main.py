#!/usr/bin/env python3
"""
main.py
Full driver for the thin airfoil solver:
- Flow field visualization (pressure & streamlines)
- Tier 1: Cl vs alpha polar comparison with XFOIL
- Tier 2: Chordwise loading ΔCp comparison with XFOIL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# Local modules
from config import *
from fourier_coefficients import fourier_coefficients
from vortex import circulation, lift_coefficient, gamma_theta
from velocity import compute_field_grid, x_of_theta
from visualize import plot_pressure_field, plot_streamfunction_heatmap, y_camber
from xfoil_parser import read_xfoil_polar, read_xfoil_cpwr

# ------------------------------------------------------------
# XFOIL data file paths (adjust to your files)
# ------------------------------------------------------------
XFOIL_POLAR_FILE = "polar.txt"                # polar data from XFOIL
XFOIL_CPWR_FILE = "cp_2412_a5.txt"            # surface pressure at α=5° (example)

# ------------------------------------------------------------
# 1. Flow field visualization (original)
# ------------------------------------------------------------
print("\n--- Flow field computation ---")
A = fourier_coefficients(M, P, ALPHA, N_FOURIER)
print(f"A0 = {A[0]:.6f}, A1 = {A[1]:.6f}, A2 = {A[2]:.6f}")

cl = lift_coefficient(A)
Gamma = circulation(V_INF, C, A)
print(f"Lift coefficient: c_l = {cl:.6f}")
print(f"Circulation: Γ = {Gamma:.6f}")

X = np.linspace(X_MIN, X_MAX, NX)
Y = np.linspace(Y_MIN, Y_MAX, NY)
X_grid, Y_grid = np.meshgrid(X, Y)

field = compute_field_grid(X_grid, Y_grid, V_INF, ALPHA, C, A,
                           include_pressure=True, include_velocity=True,
                           epsrel=1e-6)
U = field['U']
V = field['V']
Cp = field['Cp']

# Camber line points
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
plt.savefig('flow_field.png', dpi=300, bbox_inches='tight')
plt.show()
print("Flow field plot saved to flow_field.png")

# ------------------------------------------------------------
# 2. Tier 1: Polar comparison (Cl vs alpha)
# ------------------------------------------------------------
print("\n--- Tier 1: Polar comparison ---")
ALPHA_MIN_DEG = -5.0
ALPHA_MAX_DEG = 15.0
ALPHA_STEP_DEG = 1.0

alpha_deg = np.arange(ALPHA_MIN_DEG, ALPHA_MAX_DEG + ALPHA_STEP_DEG/2, ALPHA_STEP_DEG)
alpha_rad = np.deg2rad(alpha_deg)

# Thin-airfoil polar
cl_thin = []
for a_rad in alpha_rad:
    A_polar = fourier_coefficients(M, P, a_rad, N_FOURIER)
    cl_thin.append(lift_coefficient(A_polar))
cl_thin = np.array(cl_thin)

df_thin = pd.DataFrame({'alpha_deg': alpha_deg, 'Cl': cl_thin})
df_thin.to_csv("polar_thin_airfoil.csv", index=False)
print("Thin-airfoil polar saved to polar_thin_airfoil.csv")

# XFOIL polar
try:
    df_xfoil = read_xfoil_polar(XFOIL_POLAR_FILE)
    # Filter to same alpha range
    mask = (df_xfoil['alpha'] >= ALPHA_MIN_DEG) & (df_xfoil['alpha'] <= ALPHA_MAX_DEG)
    df_xfoil = df_xfoil[mask]
    print(f"XFOIL polar loaded from {XFOIL_POLAR_FILE}")
except Exception as e:
    print(f"Warning: Could not read XFOIL polar: {e}")
    df_xfoil = None

# Plot – 16:9 aspect ratio and black axes
fig, ax = plt.subplots(figsize=(10, 5.625))   # 16:9
ax.plot(alpha_deg, cl_thin, 'r-o', linewidth=2, markersize=6, label='Thin airfoil theory')
if df_xfoil is not None:
    ax.plot(df_xfoil['alpha'], df_xfoil['CL'], 'b-s', linewidth=2, markersize=6, label='XFOIL')

# Force axes to black
for spine in ax.spines.values():
    spine.set_color('black')
ax.tick_params(colors='black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

ax.set_xlabel('Angle of attack $\\alpha$ (degrees)')
ax.set_ylabel('Lift coefficient $C_l$')
ax.set_title(f'NACA {int(M*100):02d}{int(P*10):01d}??  Polar comparison')
ax.grid(True, linestyle=':', alpha=0.6)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_xticks(np.arange(ALPHA_MIN_DEG, ALPHA_MAX_DEG + 1, 1))
plt.tight_layout()
plt.savefig('polar_comparison.png', dpi=300)
plt.show()
print("Polar comparison saved to polar_comparison.png")

# ------------------------------------------------------------
# 3. Tier 2: Loading comparison (ΔCp vs x/c)
# ------------------------------------------------------------
print("\n--- Tier 2: Loading comparison ---")
# Use the same angle as in CPWR file (assume it matches)
# In this example we use ALPHA_DEG from config (5°). Adjust if needed.
ALPHA_LOAD_DEG = ALPHA_DEG
ALPHA_LOAD_RAD = np.deg2rad(ALPHA_LOAD_DEG)

# Thin-airfoil loading
A_load = fourier_coefficients(M, P, ALPHA_LOAD_RAD, N_FOURIER)
theta_min = 0.01
theta_max = np.pi - 0.01
theta = np.linspace(theta_min, theta_max, 500)
gamma = gamma_theta(theta, V_INF, A_load)
x_over_c = x_of_theta(theta, C) / C
delta_Cp_thin = 2.0 * gamma / V_INF

# Mask near leading/trailing edges
MASK_FRACTION = 0.02
mask = (x_over_c > MASK_FRACTION) & (x_over_c < (1.0 - MASK_FRACTION))
x_masked = x_over_c[mask]
delta_Cp_masked = delta_Cp_thin[mask]

# XFOIL loading from CPWR
delta_Cp_xfoil = None
try:
    upper, lower = read_xfoil_cpwr(XFOIL_CPWR_FILE)
    interp_u = interp1d(upper[:,0], upper[:,1], kind='linear', fill_value='extrapolate')
    interp_l = interp1d(lower[:,0], lower[:,1], kind='linear', fill_value='extrapolate')
    Cp_u = interp_u(x_masked)
    Cp_l = interp_l(x_masked)
    delta_Cp_xfoil = Cp_l - Cp_u
    print(f"XFOIL CPWR loaded from {XFOIL_CPWR_FILE}")
except Exception as e:
    print(f"Warning: Could not read XFOIL CPWR: {e}")

# Plot – 16:9 aspect ratio and black axes
fig, ax = plt.subplots(figsize=(10, 5.625))   # 16:9
ax.plot(x_masked, delta_Cp_masked, 'r-', linewidth=2.5, label='Thin airfoil theory')
if delta_Cp_xfoil is not None:
    ax.plot(x_masked, delta_Cp_xfoil, 'b--', linewidth=2.5, label='XFOIL')

# Force axes to black
for spine in ax.spines.values():
    spine.set_color('black')
ax.tick_params(colors='black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

ax.set_xlabel('Chordwise position $x/c$')
ax.set_ylabel('Loading $\\Delta C_p = C_{p,lower} - C_{p,upper}$')
ax.set_title(f'NACA {int(M*100):02d}{int(P*10):01d}??  Loading at $\\alpha = {ALPHA_LOAD_DEG:.0f}^\\circ$')
ax.grid(True, linestyle=':', alpha=0.6)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ymax = max(abs(delta_Cp_masked.min()), abs(delta_Cp_masked.max()))
if delta_Cp_xfoil is not None:
    ymax = max(ymax, abs(delta_Cp_xfoil.min()), abs(delta_Cp_xfoil.max()))
ymax *= 1.1
ax.set_ylim(-ymax, ymax)
plt.tight_layout()
plt.savefig(f'loading_comparison_alpha{ALPHA_LOAD_DEG:.0f}.png', dpi=300)
plt.show()
print(f"Loading comparison saved to loading_comparison_alpha{ALPHA_LOAD_DEG:.0f}.png")

# ------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------
print("\n--- Summary ---")
print(f"Thin-airfoil Cl at α = {np.rad2deg(ALPHA):.1f}°: {cl:.6f}")
print(f"Polar comparison saved in polar_comparison.png")
print(f"Loading comparison saved in loading_comparison_alpha{ALPHA_LOAD_DEG:.0f}.png")
print("All done!")