"""
streamlines.py
Streamline tracing using ODE integration.
"""

import numpy as np
from scipy.integrate import odeint
from velocity import total_velocity

def streamline_flow(state, t, V_inf, alpha, c, A):
    """
    ODE right-hand side for streamline: dx/dt = u, dy/dt = v.
    """
    x, y = state
    u, v = total_velocity(x, y, V_inf, alpha, c, A)
    return [u, v]

def compute_streamline(x0, y0, V_inf, alpha, c, A, 
                       t_max=10.0, n_steps=1000, stop_at_airfoil=True):
    """
    Integrate a single streamline from starting point (x0, y0).
    
    Parameters
    ----------
    x0, y0 : float
        Starting position
    V_inf, alpha, c, A : float/array
        Flow parameters
    t_max : float
        Integration time (determines length)
    n_steps : int
        Number of integration steps
    stop_at_airfoil : bool
        Stop if streamline hits the airfoil (|y| < 1e-4 near chord)
    
    Returns
    -------
    tuple (x_vals, y_vals, success)
        Streamline coordinates and success flag
    """
    t = np.linspace(0, t_max, n_steps)
    y0_state = [x0, y0]
    
    try:
        sol = odeint(streamline_flow, y0_state, t, 
                     args=(V_inf, alpha, c, A),
                     rtol=1e-8, atol=1e-10)
        x_vals = sol[:, 0]
        y_vals = sol[:, 1]
        
        # Check if we hit the airfoil (roughly: near chord line within [0,c])
        if stop_at_airfoil:
            for i in range(len(x_vals)):
                if 0 <= x_vals[i] <= c and abs(y_vals[i]) < 1e-4:
                    # Truncate at the crossing point
                    return x_vals[:i+1], y_vals[:i+1], True
        
        return x_vals, y_vals, True
        
    except Exception as e:
        print(f"Streamline from ({x0:.3f}, {y0:.3f}) failed: {e}")
        return np.array([x0]), np.array([y0]), False

def compute_streamlines(start_points, V_inf, alpha, c, A, **kwargs):
    """
    Compute multiple streamlines from a list of starting points.
    
    Parameters
    ----------
    start_points : list of (x0, y0)
        Starting positions
    V_inf, alpha, c, A : float/array
        Flow parameters
    **kwargs : passed to compute_streamline
    
    Returns
    -------
    list of dict
        [{'x': x_vals, 'y': y_vals, 'success': bool}, ...]
    """
    results = []
    for x0, y0 in start_points:
        x_vals, y_vals, success = compute_streamline(
            x0, y0, V_inf, alpha, c, A, **kwargs
        )
        results.append({
            'x': x_vals,
            'y': y_vals,
            'success': success,
            'start': (x0, y0)
        })
    return results