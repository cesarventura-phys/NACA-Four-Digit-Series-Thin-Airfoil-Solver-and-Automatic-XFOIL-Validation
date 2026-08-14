"""
config.py
All user-defined parameters for the thin airfoil solver.
"""

import numpy as np

# ------------------------------------------------------------
# Airfoil geometry (NACA 4-digit)
# ------------------------------------------------------------
M = 0.02          # Maximum camber (fraction of chord)
P = 0.40          # Position of max camber (fraction of chord)
C = 1.0           # Chord length (normalized)

# ------------------------------------------------------------
# Flow conditions
# ------------------------------------------------------------
ALPHA_DEG = 5.0                 # Angle of attack (degrees)
ALPHA = np.deg2rad(ALPHA_DEG)   # Convert to radians
V_INF = 1.0                     # Freestream velocity magnitude

# ------------------------------------------------------------
# Numerical parameters
# ------------------------------------------------------------
N_FOURIER = 20                  # Number of Fourier coefficients (A0..AN)

# ------------------------------------------------------------
# Grid and visualization settings
# ------------------------------------------------------------
X_MIN, X_MAX = -2.0, 3.0        # Domain bounds (in chords)
Y_MIN, Y_MAX = -2.0, 2.0
NX, NY = 80, 60               # Grid resolution

# Streamline parameters
N_STREAMLINES = 12              # Number of streamlines to plot
T_MAX = 10.0                    # Integration time
N_STEPS = 2000                  # Steps per streamline

# Masking radius around leading edge (to avoid singularity)
LE_MASK_RADIUS = 0.05 * C       # Radius in physical units