"""
vortex.py
Vortex sheet strength distribution and circulation calculations.
"""

import numpy as np

def gamma_theta(theta, V_inf, A):
    """
    Vortex strength distribution γ(θ).
    
    Parameters
    ----------
    theta : float or np.ndarray
        θ-coordinate (0 to π)
    V_inf : float
        Freestream velocity
    A : np.ndarray
        Fourier coefficients [A0, A1, ..., AN]
    
    Returns
    -------
    float or np.ndarray
        Vortex strength γ(θ)
    """
    # Extract coefficients
    A0 = A[0]
    An = A[1:]
    
    # Base term: singular at θ=0 but multiplied by sinθ later
    term0 = A0 * (1 + np.cos(theta)) / np.sin(theta)
    
    # Summation over n
    sum_terms = np.zeros_like(theta)
    for n, An_val in enumerate(An, start=1):
        sum_terms += An_val * np.sin(n * theta)
    
    return 2 * V_inf * (term0 + sum_terms)

def gamma_theta_safe(theta, V_inf, A, eps=1e-12):
    """
    γ(θ) with safety for θ near 0 or π.
    Returns the smooth product γ(θ)*sin(θ) for numerical stability.
    """
    # Use the fact that γ(θ)*sin(θ) is finite everywhere
    A0 = A[0]
    An = A[1:]
    
    product = A0 * (1 + np.cos(theta))
    for n, An_val in enumerate(An, start=1):
        product += An_val * np.sin(n * theta) * np.sin(theta)
    
    return 2 * V_inf * product

def circulation(V_inf, c, A):
    """
    Total circulation Γ = c V∞ [π A0 + (π/2) A1].
    
    Parameters
    ----------
    V_inf : float
        Freestream velocity
    c : float
        Chord length
    A : np.ndarray
        Fourier coefficients [A0, A1, ...]
    
    Returns
    -------
    float
        Total circulation per unit span
    """
    return c * V_inf * (np.pi * A[0] + 0.5 * np.pi * A[1])

def lift_coefficient(A):
    """
    Lift coefficient c_l = π(2A0 + A1).
    
    Parameters
    ----------
    A : np.ndarray
        Fourier coefficients [A0, A1, ...]
    
    Returns
    -------
    float
        Lift coefficient
    """
    return np.pi * (2*A[0] + A[1])