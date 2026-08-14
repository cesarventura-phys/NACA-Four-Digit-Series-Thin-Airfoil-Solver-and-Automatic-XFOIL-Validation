"""
fourier_coefficients.py
Exact Fourier coefficients A0..AN for a NACA 4-digit camber line.
Uses SymPy to analytically integrate the piecewise camber slope.
"""

import numpy as np
import sympy as sp

# ------------------------------------------------------------
# 1. Symbolic setup (executed at module load)
# ------------------------------------------------------------
m, p, alpha = sp.symbols('m p alpha', real=True)
theta, n = sp.symbols('theta n', real=True)

theta_p = sp.acos(1 - 2*p)          # Transition angle

# Piecewise camber slope
dy_dx1 = 2*m/p**2       * (p - (1 - sp.cos(theta))/2)
dy_dx2 = 2*m/(1-p)**2   * (p - (1 - sp.cos(theta))/2)

# ------------------------------------------------------------
# 2. A0 integral
# ------------------------------------------------------------
A0_int = (sp.integrate(dy_dx1, (theta, 0, theta_p)) +
          sp.integrate(dy_dx2, (theta, theta_p, sp.pi)))
A0_expr = alpha - (1/sp.pi) * A0_int

subs_cos_sin = {sp.cos(theta_p): 1 - 2*p,
                sp.sin(theta_p): 2*sp.sqrt(p*(1-p))}
A0_simplified = sp.simplify(A0_expr.subs(subs_cos_sin))

# ------------------------------------------------------------
# 3. An integral
# ------------------------------------------------------------
An_int = (sp.integrate(dy_dx1 * sp.cos(n*theta), (theta, 0, theta_p)) +
          sp.integrate(dy_dx2 * sp.cos(n*theta), (theta, theta_p, sp.pi)))
An_expr = (2/sp.pi) * An_int
An_simplified = sp.simplify(An_expr.subs(subs_cos_sin))

# ------------------------------------------------------------
# 4. Lambdified functions
# ------------------------------------------------------------
A0_func = sp.lambdify((m, p, alpha), A0_simplified, 'numpy')
An_func = sp.lambdify((n, m, p), An_simplified, 'numpy')

# ------------------------------------------------------------
# 5. Public API
# ------------------------------------------------------------
def fourier_coefficients(m_val, p_val, alpha_val, N=20):
    """
    Compute Fourier coefficients A0, A1, ..., AN for a NACA 4-digit airfoil.
    
    Parameters
    ----------
    m_val : float
        Maximum camber (fraction of chord)
    p_val : float
        Position of max camber (fraction of chord)
    alpha_val : float
        Angle of attack (radians)
    N : int
        Number of coefficients to compute (A0..AN)
    
    Returns
    -------
    np.ndarray
        Array of coefficients [A0, A1, ..., AN]
    """
    if m_val == 0.0:
        A0 = alpha_val
        An = [0.0] * N
        return np.array([A0] + An, dtype=float)
    
    A0 = A0_func(m_val, p_val, alpha_val)
    # An_func is a lambdified sympy Piecewise compiled to numpy.select, which
    # evaluates every branch (including the n=1 branch's 0/0 division) before
    # discarding the unused ones. The selected result is correct; only the
    # discarded branch produces nan, so it's safe to suppress the warning here.
    with np.errstate(divide='ignore', invalid='ignore'):
        An = [An_func(k, m_val, p_val) for k in range(1, N + 1)]
    return np.array([A0] + An, dtype=float)