"""
visualize.py
Plotting utilities for the thin airfoil solver.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Airfoil geometry
# ------------------------------------------------------------
def y_camber(x, c, m, p):
    """Camber height at chordwise position x."""
    xc = x / c
    if isinstance(xc, np.ndarray):
        y = np.zeros_like(x)
        mask1 = xc <= p
        mask2 = xc > p
        y[mask1] = c * m / p**2 * (2*p*xc[mask1] - xc[mask1]**2)
        y[mask2] = c * m / (1-p)**2 * ((1 - 2*p) + 2*p*xc[mask2] - xc[mask2]**2)
        return y
    else:
        if xc <= p:
            return c * m / p**2 * (2*p*xc - xc**2)
        else:
            return c * m / (1-p)**2 * ((1 - 2*p) + 2*p*xc - xc**2)

# ------------------------------------------------------------
# Pressure field (no zero‑contour line)
# ------------------------------------------------------------
def plot_pressure_field(X, Y, Cp, ax=None, mask_radius=0.05):
    if ax is None:
        ax = plt.gca()
    r = np.sqrt(X**2 + Y**2)
    Cp_masked = np.ma.masked_where(r < mask_radius, Cp)
    levels = np.linspace(-1.5, 1.0, 30)
    cf = ax.contourf(X, Y, Cp_masked, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=ax, label='$C_p$')
    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$y/c$')
    ax.set_aspect('equal')
    return cf

# ------------------------------------------------------------
# Stream function – compute from velocity field
# ------------------------------------------------------------
def compute_streamfunction(X, Y, U, V):
    dy = Y[1,0] - Y[0,0]
    dx = X[0,1] - X[0,0]
    ny, nx = X.shape
    ψ = np.zeros_like(X)

    # Step 1: march across the bottom row using dψ/dx = -v
    for i in range(1, nx):
        ψ[0,i] = ψ[0,i-1] - 0.5 * (V[0,i-1] + V[0,i]) * dx

    # Step 2: march up each column using dψ/dy = u
    for i in range(nx):
        for j in range(1, ny):
            ψ[j,i] = ψ[j-1,i] + 0.5 * (U[j-1,i] + U[j,i]) * dy
    return ψ

def plot_streamfunction_heatmap(X, Y, U, V, ax=None, mask_radius=0.05,
                                draw_contour_lines=True, line_color='k', line_width=0.8):
    """
    Filled contour of stream function + optional black contour lines.
    """
    if ax is None:
        ax = plt.gca()
    ψ = compute_streamfunction(X, Y, U, V)
    r = np.sqrt(X**2 + Y**2)
    ψ_masked = np.ma.masked_where(r < mask_radius, ψ)
    
    # Heatmap (filled contours)
    levels_fill = np.linspace(ψ.min(), ψ.max(), 60)
    cf = ax.contourf(X, Y, ψ_masked, levels=levels_fill, cmap='inferno', extend='both')
    plt.colorbar(cf, ax=ax, label='Stream function $\psi$')
    
    # Overlay black contour lines (streamlines)
    if draw_contour_lines:
        # Use a smaller number of levels for clarity
        levels_lines = np.linspace(ψ.min(), ψ.max(), 20)
        ax.contour(X, Y, ψ_masked, levels=levels_lines,
                   colors=line_color, linewidths=line_width, alpha=0.7)
    
    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$y/c$')
    ax.set_aspect('equal')
    return cf