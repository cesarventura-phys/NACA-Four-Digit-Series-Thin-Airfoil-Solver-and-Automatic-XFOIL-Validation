"""
xfoil_parser.py
Robust parsers for XFOIL output files.
Handles polar (pacc) and surface pressure (CPWR) files with headers.
"""

import numpy as np
import pandas as pd
import re

def read_xfoil_polar(filename):
    """
    Read XFOIL polar file and return a pandas DataFrame.
    Skips header lines and finds the column header row.
    Columns: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the line that starts with "  alpha" or "alpha" (column header)
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\balpha\b', line, re.IGNORECASE) and re.search(r'\bCL\b', line, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find column header in polar file")
    
    # Data starts after the header line
    data_lines = []
    for line in lines[header_idx+1:]:
        # Skip lines that are empty or contain only dashes
        if line.strip() == '' or line.strip().startswith('---'):
            continue
        # Check if line contains numbers
        parts = line.split()
        if len(parts) >= 6 and all(re.match(r'^-?\d*\.?\d+', p) for p in parts[:6]):
            data_lines.append(line.strip())
    
    # Parse data
    data = [list(map(float, line.split())) for line in data_lines]
    df = pd.DataFrame(data, columns=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
    return df

def read_xfoil_cpwr(filename):
    """
    Read XFOIL CPWR file and return two numpy arrays:
        upper: (x, Cp) for upper surface, ordered LE→TE
        lower: (x, Cp) for lower surface, ordered LE→TE
    
    The file contains two blocks (upper and lower). Each block has a header
    line starting with '#', then data columns: x, y, Cp.
    We separate surfaces by the sign of y (positive = upper, negative = lower).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract data lines that contain three numbers (x, y, Cp)
    data_points = []
    for line in lines:
        parts = line.split()
        # Skip header lines starting with '#', or lines with text
        if not parts:
            continue
        if parts[0].startswith('#') or parts[0].startswith('NACA') or parts[0].startswith('Alfa'):
            continue
        # Try to parse three numbers
        if len(parts) >= 3:
            try:
                x = float(parts[0])
                y = float(parts[1])
                cp = float(parts[2])
                data_points.append((x, y, cp))
            except ValueError:
                continue
    
    if not data_points:
        raise ValueError("No numeric data found in CPWR file")
    
    # Separate by y > 0 (upper) and y < 0 (lower)
    upper_pts = [(x, cp) for x, y, cp in data_points if y > 0]
    lower_pts = [(x, cp) for x, y, cp in data_points if y < 0]
    
    if not upper_pts or not lower_pts:
        raise ValueError("Could not separate upper and lower surfaces (check y values)")
    
    # Sort by x ascending (LE→TE)
    upper_pts.sort(key=lambda p: p[0])
    lower_pts.sort(key=lambda p: p[0])
    
    upper = np.array(upper_pts)
    lower = np.array(lower_pts)
    return upper, lower