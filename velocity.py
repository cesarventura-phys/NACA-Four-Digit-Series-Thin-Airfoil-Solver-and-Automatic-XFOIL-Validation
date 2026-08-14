"""
velocity.py
Velocity field computation: induced, total, and pressure coefficient.
"""

import numpy as np
from scipy.integrate import quad
from vortex import gamma_theta_safe

# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def x_of_theta(theta, c=1.0):
    """Convert θ to chordwise coordinate x."""
    return 0.5 * c * (1 - np.cos(theta))

def theta_of_x(x, c=1.0):
    """Convert x to θ."""
    return np.arccos(1 - 2*x/c)

# ------------------------------------------------------------
# Induced velocity (single point)
# ------------------------------------------------------------
def induced_velocity(x, y, V_inf, c, A, epsabs=1e-8, epsrel=1e-6):
    """
    Compute induced velocity at a field point (x, y) using Biot-Savart.
    """
    # If exactly on the chord line, perturb y slightly to avoid singularity
    if abs(y) < 1e-14:
        y = 1e-14 * (1.0 if y >= 0 else -1.0)
    
    def integrand_u(theta):
        xi = x_of_theta(theta, c)
        r2 = (x - xi)**2 + y**2
        gamma_sin = gamma_theta_safe(theta, V_inf, A)
        return gamma_sin * y / r2 * (c/2) / (2*np.pi)
    
    def integrand_v(theta):
        xi = x_of_theta(theta, c)
        r2 = (x - xi)**2 + y**2
        gamma_sin = gamma_theta_safe(theta, V_inf, A)
        return -gamma_sin * (x - xi) / r2 * (c/2) / (2*np.pi)
    
    u, _ = quad(integrand_u, 0, np.pi, epsabs=epsabs, epsrel=epsrel)
    v, _ = quad(integrand_v, 0, np.pi, epsabs=epsabs, epsrel=epsrel)
    return u, v

# ------------------------------------------------------------
# Total velocity
# ------------------------------------------------------------
def total_velocity(x, y, V_inf, alpha, c, A, **quad_kwargs):
    """
    Total velocity = freestream + induced.
    
    Parameters
    ----------
    x, y : float
        Field point coordinates
    V_inf : float
        Freestream velocity magnitude
    alpha : float
        Angle of attack (radians)
    c : float
        Chord length
    A : np.ndarray
        Fourier coefficients
    
    Returns
    -------
    tuple (u_total, v_total)
    """
    u_ind, v_ind = induced_velocity(x, y, V_inf, c, A, **quad_kwargs)
    u = V_inf * np.cos(alpha) + u_ind
    v = V_inf * np.sin(alpha) + v_ind
    return u, v

# ------------------------------------------------------------
# Pressure coefficient
# ------------------------------------------------------------
def pressure_coefficient(x, y, V_inf, alpha, c, A, **quad_kwargs):
    """
    Pressure coefficient C_p = 1 - (u² + v²)/V∞².
    """
    u, v = total_velocity(x, y, V_inf, alpha, c, A, **quad_kwargs)
    V2 = u**2 + v**2
    return 1.0 - V2 / V_inf**2

# ------------------------------------------------------------
# Vectorized field computation (for grid evaluation)
# ------------------------------------------------------------
def compute_field_grid(X, Y, V_inf, alpha, c, A, 
                       include_pressure=True, include_velocity=True,
                       **quad_kwargs):
    """
    Compute velocity and/or pressure on a 2D grid.
    
    Parameters
    ----------
    X, Y : 2D np.ndarray
        Grid coordinates
    V_inf, alpha, c, A : float/array
        Flow parameters
    include_pressure, include_velocity : bool
        What to compute
    
    Returns
    -------
    dict
        {'U': u_grid, 'V': v_grid, 'Cp': cp_grid} (as applicable)
    """
    result = {}
    shape = X.shape
    
    if include_velocity:
        U = np.zeros(shape)
        V = np.zeros(shape)
    
    if include_pressure:
        Cp = np.zeros(shape)
    
    # Loop over grid points (PoC: simple but slow)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = X[i, j]
            y = Y[i, j]
            
            if include_velocity:
                u, v = total_velocity(x, y, V_inf, alpha, c, A, **quad_kwargs)
                U[i, j] = u
                V[i, j] = v
            
            if include_pressure:
                if include_velocity:
                    Cp[i, j] = 1.0 - (u**2 + v**2) / V_inf**2
                else:
                    u, v = total_velocity(x, y, V_inf, alpha, c, A, **quad_kwargs)
                    Cp[i, j] = 1.0 - (u**2 + v**2) / V_inf**2
    
    if include_velocity:
        result['U'] = U
        result['V'] = V
    if include_pressure:
        result['Cp'] = Cp
    
    return result